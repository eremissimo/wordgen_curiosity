import torch
import torch.nn as nn
import torch.nn.functional as ff
import torch.optim as optim
import einops
from torch.distributions import Categorical
from typing import Iterable, Optional


class CharTransformer(nn.Module):
    def __init__(self, n_token: int, model_cfg: dict):
        super().__init__()
        self.max_word_len = model_cfg["max_word_len"]
        self.content_embd = nn.Embedding(n_token, model_cfg["d_model"], padding_idx=0)
        self.positional_embd = nn.Embedding(model_cfg["max_word_len"], model_cfg["d_model"])
        layer = nn.TransformerEncoderLayer(model_cfg["d_model"],
                                           model_cfg["n_head"],
                                           model_cfg["dim_feedforward"],
                                           model_cfg["dropout"],
                                           activation=ff.relu,
                                           batch_first=True,
                                           norm_first=False)    # [batch, seq, feat]
        self.decoder = nn.TransformerEncoder(layer, model_cfg["num_layers"], norm=None)
        self.decoder_head = nn.Linear(model_cfg["d_model"], n_token)

    def forward(self, x):
        pad_mask = (x == 0)
        x = self.content_embd(x) + self.positional_embd(positions(x))     # [batch, seq, d_model]
        x = self.decoder(x, mask=generate_square_subsequent_mask(x.shape[1], x.device),
                         src_key_padding_mask=pad_mask,
                         is_causal=True)
        x = self.decoder_head(x)
        return x

    @property
    def device(self):
        return next(self.content_embd.parameters()).device

    def predict_sample(self, x):
        # predict next tokens by sampling the probability defined by logits
        logits = self.forward(x)[..., -1:, :]   # [batch, 1, n_tokens]
        dist = Categorical(logits=logits)
        tokens = dist.sample()
        return tokens, dist.log_prob(tokens)

    def predict_argmax(self, x):
        # predict in argmax mode for RL baselines (self-critical sequence training)
        logits = self.forward(x)[..., -1:, :]       # [batch, 1, n_tokens]
        tokens = logits.argmax(dim=-1)
        return tokens

    def generate(self, batch_size: int, n: Optional[int] = None):
        self.eval()
        if n is None:
            n = self.max_word_len
        elif n > self.max_word_len:
            raise ValueError(f"Cannot generate more than {self.max_word_len=} tokens")
        device = self.device
        token = torch.ones((batch_size, 1), dtype=torch.long, device=device)   # begin word token
        output = token.clone()
        for i in range(1, n):
            token, _ = self.predict_sample(output)
            output = torch.cat((output, token), dim=-1)
        return output


def positions(x):
    # for pos embedding
    # x.shape = [batch, seq]
    return torch.arange(x.shape[1], device=x.device, requires_grad=False)


def generate_square_subsequent_mask(size, device):
    return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)


class DummyTransformerEncoder(nn.Module):
    """A module for 'teacher' and 'student' models in curiosity class."""
    def __init__(self, n_token: int, d_model: int, n_output: int, n_head: int = 4, num_layers: int = 3):
        super().__init__()
        max_len = 40
        self.content_embd = nn.Embedding(n_token, d_model, padding_idx=0)
        self.positional_embd = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model,
                                           n_head,
                                           dim_feedforward=4*d_model,
                                           dropout=0.0,
                                           activation=ff.relu,
                                           batch_first=True,
                                           norm_first=False)    # [batch, seq, feat]
        self.decoder = nn.TransformerEncoder(layer, num_layers, norm=None)
        self.decoder_head = nn.Linear(d_model, n_output)

    def forward(self, x):
        pad_mask = (x == 0)
        x = self.content_embd(x) + self.positional_embd(positions(x))     # [batch, seq, d_model]
        x = self.decoder(x, mask=generate_square_subsequent_mask(x.shape[1], x.device),
                         src_key_padding_mask=pad_mask,
                         is_causal=True)
        x = self.decoder_head(x)
        return x


class CuriosityReward(nn.Module):
    """A distillation based curiosity. The agent gets more reward for getting to unfamiliar state
    (i.e. when there is a discrepancy between teacher and student model outputs)"""
    def __init__(self, teacher: nn.Module, student: nn.Module, lr: float = 1e-3,
                 temperature: float | Iterable[float] = 1., scale: float = 1.):
        super().__init__()
        self._teacher = teacher
        self._student = student
        self._optimizer = optim.Adam(self._student.parameters(), lr=lr)
        if not isinstance(temperature, Iterable):
            temperature = (temperature, temperature)
        self._t_temp, self._s_temp = temperature    # hyperparameter: sharpness of softmax distribution
        self.scale = scale                          # hyperparameter: scales up the resulting reward

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Given the state perform a single curiosity reward calculation (distill_loss) with a single backprop step"""
        # assuming state is on the same device as teacher and student models
        with torch.no_grad():
            targets = ff.softmax(self._teacher(states) * self._t_temp, dim=-1)
        self._optimizer.zero_grad()
        inputs = self._student(states) * self._s_temp
        reduce_dims = tuple(range(int(targets.ndim > 2)+1, targets.ndim))    # (0,) for unbatched and (1,...,n) for batched
        distill_loss = ff.cross_entropy(inputs.transpose(-2, -1), targets.transpose(-2, -1))
        distill_metric = ff.mse_loss(inputs.softmax(-1), targets, reduction="none").mean(dim=reduce_dims)
        distill_loss.backward()
        self._optimizer.step()
        return distill_metric.detach() * self.scale

    def calibrate_scale(self, states: torch.Tensor, targ_reward: float):
        """Setting self.scale so that the output curiosity reward is approximately equal targ_reward"""
        with torch.no_grad():
            targets = ff.softmax(self._teacher(states) * self._t_temp, dim=-1)
            inputs = ff.softmax(self._student(states) * self._s_temp, dim=-1)
        loss = ff.mse_loss(inputs, targets).item()
        self.scale = (targ_reward/loss)

    @property
    def device(self):
        return next(self._teacher.parameters()).device


class CuriosityRewardTransformer(CuriosityReward):
    def __init__(self, n_token: int, lr: float = 1e-3,
                 temperature: float | Iterable[float] = 1., scale: float = 1.):
        out_size = 40
        d_model = 8
        teacher = DummyTransformerEncoder(n_token, d_model, out_size, 4, 3)
        student = DummyTransformerEncoder(n_token, d_model, out_size, 4, 3)
        super().__init__(teacher, student, lr=lr, temperature=temperature, scale=scale)


class WordReward:
    """Assigning reward when a generated word is present in the dictionary. Using TokenTrie for prefix search."""
    def __init__(self, token_trie, status_reward_mapping):
        self._token_trie = token_trie
        padding_reward = 0.0
        _remapping_values = [padding_reward] + [status_reward_mapping[k] for k in
                                                   ["nonword_char", "word_char", "test_word_char", "train_word_char"]]
        self._reward_mapping_values = torch.tensor(_remapping_values, dtype=torch.get_default_dtype())
        self._full_word_reward = status_reward_mapping["full_word"]

    def __call__(self, token_words: torch.Tensor) -> torch.Tensor:
        maxn = token_words.shape[1]
        device = token_words.device
        token_words = token_words.cpu()
        values, is_full_word = self._token_trie.word_values(token_words)
        filled_pad_mask = self._fill_holes_in_paddings_and_invert(values == -1)
        values = torch.where(filled_pad_mask, values, -1)
        values = self._reward_mapping_values[values+1]
        values = values @ torch.tril(torch.ones(maxn, maxn))    # Q(s,a) calculation as reverse cumsum
        # taking care of the final reward at the end of the word (or episode in the terms of RL)
        full_word_reward = is_full_word * (self._full_word_reward - values[:, 0])
        values += filled_pad_mask * full_word_reward.unsqueeze(1)
        return values.to(device)

    @staticmethod
    def _fill_holes_in_paddings_and_invert(mask: torch.Tensor) -> torch.Tensor:
        # [[0, 0, 1, 0, 1, 0, 0, 1, 1]] -> [[1, 1, 0, 0, 0, 0, 0, 0, 0]]
        return torch.arange(mask.shape[1], dtype=torch.short).unsqueeze(0) < mask.short().argmax(dim=1, keepdim=True)


def save_checkpoint(model, path, optimizer=None):
    device = model.device
    model.cpu()
    torch.save(model.state_dict(), path)
    model.to(device)


def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path))


if __name__ == "__main__":
    from data import get_data

    data_cfg = {"path": "dictionary.csv",
                "target_pos": "v. t.",
                "train_test_proportion": (1, 2),
                "batch_size": 4,
                "num_workers": 0}
    model_cfg = {"max_word_len": 30,
                 "d_model": 8,
                 "n_head": 2,
                 "dim_feedforward": 20,
                 "dropout": 0.1,
                 "num_layers": 2}
    rl_remap = {
            "nonword_char": -1.0,
            "word_char": 0.0,
            "test_word_char": 0.0,
            "train_word_char": 1.0,
            "full_word": 35.0
        }
    dataloader, _, tokenizer, rl_trie, _ = get_data(data_cfg)
    model = CharTransformer(tokenizer.n_tokens, model_cfg)
    inp, targ = next(iter(dataloader))
    out = model(inp)
    print(targ.shape)
    print(out.shape)
    pred1 = model.predict_argmax(inp)
    pred2 = model.predict_sample(inp)
    print("pred_argmax: ", pred1.shape)
    print("pred_sample: ", pred2[0].shape)
    print("pred_sample_logprobs: ", pred2[1].shape)

    gen = model.generate(10, 20).numpy().tolist()
    print(tokenizer.decode(gen))

    print("\n\n *********** \n\n")
    print("Curiosity")
    states = inp
    curiosity = CuriosityRewardTransformer(tokenizer.n_tokens, scale=1000., lr=1e-2, temperature=(20., 1.))
    curiosity.calibrate_scale(states, 1.)
    print("surprised: ", curiosity(states))
    for _ in range(100):
        curiosity(states)
    print("bored: ", curiosity(states))
    states2 = states
    states2[:, 5:] = torch.randint(0, 30, (4, 21))
    print("partly bored: ", curiosity(states2))

    print("\n\n *********** \n\n")
    print("WordReward")
    rewards = WordReward(rl_trie, rl_remap)
    print(rewards(inp))


    print("woah!")
