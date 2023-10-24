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
        x = self.content_embd(x) + self.positional_embd(self._positions(x))     # [batch, seq, d_model]
        x = self.decoder(x, mask=self._generate_square_subsequent_mask(x.shape[1], x.device),
                         src_key_padding_mask=pad_mask,
                         is_causal=True)
        x = self.decoder_head(x)
        return x

    def _positions(self, x):
        # for pos embedding
        # x.shape = [batch, seq]
        return torch.arange(x.shape[1], device=x.device, requires_grad=False)

    @staticmethod
    def _generate_square_subsequent_mask(size, device):
        return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)

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


class DummyGRUModel(nn.Module):
    """A module for 'teacher' and 'student' models in curiosity class."""
    def __init__(self, n_token: int, input_size: int, output_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.legs = nn.Embedding(n_token, input_size, padding_idx=0)
        self.body = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(num_layers*hidden_size, output_size)
        self.h0 = nn.Parameter(torch.randn((num_layers, hidden_size)), requires_grad=True)

    def forward(self, x):
        # assuming x is a LongTensor of token indices of shape [batch, seq]
        x = self.legs(x)
        _, x = self.body(x, self._batched_h0(x))
        x = einops.rearrange(x, "nlayers ... hidden -> ... (nlayers hidden)")   # [batch, feat] or [feat]
        x = self.head(x)
        return x

    def _batched_h0(self, x):
        if x.ndim == 3:
            # batched input
            return einops.repeat(self.h0, "nlayers hidden -> nlayers nbatch hidden", nbatch=x.shape[0])
        return self.h0


class CuriosityReward(nn.Module):
    """A distillation based curiosity. The agent gets more reward for getting to unknown state
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
        inputs = ff.softmax(self._student(states) * self._s_temp, dim=-1)
        reduce_dims = tuple(range(int(targets.ndim > 1), targets.ndim))    # (0,) for unbatched and (1,...,n) for batched
        distill_losses = ff.mse_loss(inputs, targets, reduction="none").mean(dim=reduce_dims)
        distill_losses.mean().backward()
        self._optimizer.step()
        return distill_losses.detach() * self.scale


class CuriosityRewardGRU(CuriosityReward):
    def __init__(self, n_token: int, lr: float = 1e-3,
                 temperature: float | Iterable[float] = 1., scale: float = 1.):
        out_size = 40
        teacher = DummyGRUModel(n_token, 6, out_size, 25, 3)
        student = DummyGRUModel(n_token, 6, out_size, 25, 3)
        super().__init__(teacher, student, lr=lr, temperature=temperature, scale=scale)


def save_checkpoint(model, path, optimizer=None):
    torch.save(model.cpu().state_dict(), path)


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
    dataloader, _, tokenizer, _ = get_data(data_cfg)
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
    curiosity = CuriosityRewardGRU(tokenizer.n_tokens, scale=1000., temperature=(20., 1.))
    print("surprised: ", curiosity(states))
    for i in range(100):
        curiosity(states)
    print("bored: ", curiosity(states))

    print("woah!")
