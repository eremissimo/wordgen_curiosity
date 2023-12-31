import torch
import torch.nn as nn
import torch.nn.functional as ff
import torch.optim as optim
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

    def predict_sample_train(self, x):
        # predict next tokens by sampling the probability defined by logits
        logits = self.forward(x)[..., -1:, :]   # [batch, 1, n_tokens]
        dist = Categorical(logits=logits)
        tokens = dist.sample()
        return tokens, dist.log_prob(tokens), dist.entropy()

    def predict_argmax(self, x):
        # predict in argmax mode for RL baselines (self-critical sequence training)
        logits = self.forward(x)[..., -1:, :]       # [batch, 1, n_tokens]
        tokens = logits.argmax(dim=-1)
        return tokens

    def predict_sample(self, x):
        # predict in sampling mode for evaluation
        logits = self.forward(x)[..., -1:, :]   # [batch, 1, n_tokens]
        dist = Categorical(logits=logits)
        tokens = dist.sample()
        return tokens

    def generate_sample_train(self, batch_size: int, n: Optional[int] = None):
        # this method is used in rl training
        token, output, n, device = self._prepare_gen(batch_size, n)
        log_probs = torch.zeros((batch_size, 1), dtype=torch.float32, device=device, requires_grad=False)
        entropy = log_probs.clone()
        for i in range(1, n):
            token, logp, entr = self.predict_sample_train(output)
            output = torch.cat((output, token), dim=-1)
            log_probs = torch.cat((log_probs, logp), dim=-1)
            entropy = torch.cat((entropy, entr), dim=-1)
        return output, log_probs, entropy

    @torch.no_grad()
    def generate_sample(self, batch_size: int, n: Optional[int] = None):
        # this method is used for sampling at evaluation
        token, output, n, _ = self._prepare_gen(batch_size, n)
        for i in range(1, n):
            token = self.predict_sample(output)
            output = torch.cat((output, token), dim=-1)
        return output

    @torch.no_grad()
    def generate_argmax(self, batch_size: int, n: Optional[int] = None):
        # this method is not used for training (unlike generate_sample_train) hence the no_grad deco
        token, output, n, _ = self._prepare_gen(batch_size, n)
        for i in range(1, n):
            token = self.predict_argmax(output)
            output = torch.cat((output, token), dim=-1)
        return output

    def _prepare_gen(self, batch_size, n):
        # common code for generate_xxx functions
        if n is None:
            n = self.max_word_len
        elif n > self.max_word_len:
            raise ValueError(f"Cannot generate more than {self.max_word_len=} tokens")
        device = self.device
        token = torch.ones((batch_size, 1), dtype=torch.long, device=device)  # begin word token
        output = token.clone()
        return token, output, n, device

    def freeze_n_layers(self, n: int):
        self.content_embd.requires_grad_(False)
        self.positional_embd.requires_grad_(False)
        for layer in self.decoder.layers[:n]:
            layer.requires_grad_(False)

    def freeze(self):
        self.requires_grad_(False)

    def unfreeze(self):
        self.requires_grad_(True)


def positions(x):
    # for pos embedding
    # x.shape = [batch, seq]
    return torch.arange(x.shape[1], device=x.device, requires_grad=False)


def generate_square_subsequent_mask(size, device):
    return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)


class DummyTransformerEncoder(nn.Module):
    """A module for 'teacher' and 'student' models in curiosity class."""
    # TODO: make this a base class for CharTransformer to keep it DRY
    def __init__(self, n_token: int, d_model: int, n_output: int, n_head: int = 4, num_layers: int = 3):
        super().__init__()
        max_len = 32
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
    def __init__(self, mastermind: nn.Module, teacher: nn.Module, student: nn.Module, lr: float = 1e-3,
                 temperature: float | Iterable[float] = 1., scale: float = 1., lr_t: Optional[float] = None):
        super().__init__()
        self._mastermind = mastermind
        self._teacher = teacher
        self._student = student
        self._s_optimizer = optim.AdamW(self._student.parameters(), lr=lr, weight_decay=1e-4)
        lr_t = lr_t or lr/10.
        self._t_optimizer = optim.AdamW(self._teacher.parameters(), lr=lr_t, weight_decay=1e-4)
        if not isinstance(temperature, Iterable):
            temperature = (temperature, temperature, temperature)
        self._m_temp, self._t_temp, self._s_temp = temperature    # hyperparameter: sharpness of softmax distribution
        self.scale = scale                          # hyperparameter: scales up the resulting reward
        self.register_buffer("_targ_reward", torch.tensor([float("inf")], device=self.device))

    def forward(self, states: torch.Tensor, optim_step=True) -> torch.Tensor:
        """Given the state perform a single curiosity reward calculation (distill_loss) with a single backprop step"""
        # assuming state is on the same device as teacher and student models
        if optim_step:
            with torch.no_grad():
                t_targets = ff.softmax(self._mastermind(states) * self._m_temp, dim=-1)
            self._s_optimizer.zero_grad()
            self._t_optimizer.zero_grad()
        t_inputs = self._teacher(states) * self._t_temp
        s_targets = t_inputs.detach().softmax(-1)
        s_inputs = self._student(states) * self._s_temp
        reduce_dims = tuple(range(int(s_targets.ndim > 2)+1, s_targets.ndim))    # (0,) for unbatched and (1,...,n) for batched
        mask = self._get_pad_mask(states)
        distill_metric = self.masked_loss(ff.mse_loss,
                                          s_inputs.softmax(-1),
                                          s_targets,
                                          mask.unsqueeze(-1),
                                          reduce_dims=reduce_dims).detach()
        if optim_step:
            distill_loss = self.masked_loss(ff.cross_entropy,
                                            s_inputs.transpose(-2, -1),
                                            s_targets.transpose(-2, -1),
                                            mask,
                                            reduce_dims="all")
            distill_loss.backward()
            self._s_optimizer.step()
            drift_loss = self.masked_loss(ff.cross_entropy,
                                          t_inputs.transpose(-2, -1),
                                          t_targets.transpose(-2, -1),
                                          mask,
                                          reduce_dims="all")
            drift_loss.backward()
            self._t_optimizer.step()
        return torch.minimum(distill_metric * self.scale, self._targ_reward)

    def calibrate_scale(self, states: torch.Tensor, targ_reward: float):
        """Setting self.scale so that the output curiosity reward is approximately equal targ_reward"""
        with torch.no_grad():
            targets = ff.softmax(self._teacher(states) * self._t_temp, dim=-1)
            inputs = ff.softmax(self._student(states) * self._s_temp, dim=-1)
        loss = self.masked_loss(ff.mse_loss, inputs, targets, self._get_pad_mask(states).unsqueeze(-1))
        self.scale = (targ_reward/loss)
        self._targ_reward[0] = targ_reward

    @staticmethod
    def _get_pad_mask(states, pad_idx=0):
        return states != pad_idx

    @staticmethod
    def masked_loss(loss_fn, inp, targ, mask, reduce_dims: str | tuple = "all"):
        loss = loss_fn(inp, targ, reduction="none") * mask
        if reduce_dims == "all":
            loss = loss.mean()
        elif isinstance(reduce_dims, tuple):
            loss = loss.mean(dim=reduce_dims)
        return loss

    @property
    def device(self):
        return next(self._teacher.parameters()).device


class CuriosityRewardTransformer(CuriosityReward):
    def __init__(self, n_token: int, lr: float = 1e-3,
                 temperature: float | Iterable[float] = 1., scale: float = 1., lr_t: Optional[float] = None):
        out_size = 40
        d_model = 8
        mastermind = DummyTransformerEncoder(n_token, d_model, out_size, 4, 2)
        teacher = DummyTransformerEncoder(n_token, d_model, out_size, 4, 2)
        student = DummyTransformerEncoder(n_token, d_model, out_size, 4, 2)
        super().__init__(mastermind, teacher, student, lr=lr, temperature=temperature, scale=scale, lr_t=lr_t)


class WordReward(nn.Module):
    """Assigning reward when a generated word is present in the dictionary. Using TokenTrie for prefix search."""
    def __init__(self, token_trie, status_reward_mapping: dict):
        super().__init__()
        self._token_trie = token_trie
        padding_reward = 0.0
        _remapping_values = [padding_reward] + [status_reward_mapping[k] for k in
                                                   ["nonword_char", "word_char", "test_word_char", "train_word_char"]]
        _reward_mapping_values = torch.tensor(_remapping_values, dtype=torch.float32)
        self.register_buffer("_reward_mapping_values", _reward_mapping_values)
        self._full_word_reward = status_reward_mapping["full_word"] or 0.0

    def forward(self, token_words: torch.Tensor) -> torch.Tensor:
        end_token = 2
        maxn = token_words.shape[1]
        device = token_words.device
        end_idxs = (token_words == end_token).nonzero()
        # token trie is cpu-only thing
        values, is_full_word = (x.to(device) for x in self._token_trie.word_values(token_words.cpu()))
        # all the following computations are performed on device
        values = self._reward_mapping_values[values+1]
        # taking care of the final reward at the end of the word (or episode in the terms of RL)
        if abs(self._full_word_reward) > 1e-3:
            full_word_reward = is_full_word * self._full_word_reward
            values[end_idxs[:, 0], end_idxs[:, 1]] += full_word_reward[end_idxs[:, 0]]
        return values


class WeightedSumRewards(nn.Module):
    def __init__(self, reward: WordReward, curiosity: CuriosityReward, coeff: float):
        super().__init__()
        self._reward = reward
        self._curiosity = curiosity
        self._coeff_src = coeff

    def forward(self, states: torch.Tensor, optim_step=True) -> torch.Tensor:
        rew = self._reward(states)
        if self._coeff_src:
            rew += self._coeff_src * self._curiosity(states, optim_step=optim_step)
        return rew

    def update_weight(self, val: float):
        self._coeff_src = float(val)


class QValueAggregator(nn.Module):
    """Calculates Q-values as reverse cumulative reward"""
    def __init__(self, reward_module: nn.Module, max_len: int):
        super().__init__()
        self._reward = reward_module
        # a linear operator matrix of reverse cumsum
        rev_cumsum_op = torch.tril(torch.ones((max_len, max_len), dtype=torch.float32))
        self.register_buffer("_rev_cumsum_op", rev_cumsum_op)

    def forward(self, states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = self._reward(states, *args, **kwargs)
        values = values @ self._rev_cumsum_op
        return values


def save_checkpoint(model, path, optimizer=None):
    device = model.device
    model.cpu()
    torch.save(model.state_dict(), path)
    model.to(device)


def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path, map_location=model.device))

