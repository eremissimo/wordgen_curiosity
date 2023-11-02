import json
import re
import collections
import torch
import torch.nn as nn
import torch.nn.functional as ff
import torch.optim as optim
from typing import List, Optional, Callable

from data import get_data, TokenTrie, CharTokenizer
from model import CharTransformer, save_checkpoint, load_checkpoint, WeightedSumRewards, \
                   WordReward, CuriosityRewardTransformer, QValueAggregator


def pretrain(config: dict):
    """A 'teacher forcing' pretraining of the model on a set of words"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg = config["data_cfg"]
    model_cfg = config["model_cfg"]
    pretrain_cfg = config["pretrain_cfg"]
    pretrain_dataloader, pretrain_val_dataloader, tokenizer, token_word_checker, str_word_checker = get_data(data_cfg)
    model = CharTransformer(tokenizer.n_tokens, model_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=pretrain_cfg["lr"], weight_decay=pretrain_cfg["l2reg"])
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                            lambda ep: min(1, pretrain_cfg["lr_gamma"]**(ep - pretrain_cfg["warmup_epochs"])))
    if (ckpt_path := pretrain_cfg["checkpoint"]) is not None:
        load_checkpoint(model, ckpt_path)       # TODO: add possibility to save/load models together with optimizers
    for epoch in range(pretrain_cfg["epochs"]):
        train_loss = train_epoch(model, optimizer, pretrain_dataloader, device)
        val_loss = val_epoch(model, pretrain_val_dataloader, device)
        scheduler.step()
        print(f"epoch {epoch}: train: {train_loss} | val: {val_loss}")
        if (epoch + 1) % 20 == 0:
            save_checkpoint(model, f"./checkpoints/ChTrans_2_{val_loss:.4f}.pt")
    save_checkpoint(model, "./checkpoints/ChTrans_2_pretrained.pt")
    return model, tokenizer, token_word_checker, str_word_checker


def train_epoch(model, optimizer, dataloader, device):
    """One epoch of training"""
    model.train()
    train_loss = torch.tensor(0., device=device, requires_grad=False)
    n = len(dataloader)
    for batch in dataloader:
        inputs, targets = (x.to(device) for x in batch)    # [batch, seq, n_tokens]
        optimizer.zero_grad()
        logits = model(inputs)
        loss = ff.cross_entropy(logits.transpose(-1, -2), targets)
        loss.backward()
        optimizer.step()
        train_loss += (loss.detach() / n)
    return train_loss.item()


@torch.no_grad()
def val_epoch(model, dataloader, device):
    """Validation step"""
    model.eval()
    val_loss = torch.tensor(0., device=device)
    n = len(dataloader)
    for batch in dataloader:
        inputs, targets = (x.to(device) for x in batch)
        logits = model(inputs)
        loss = ff.cross_entropy(logits.transpose(-1,-2), targets)
        val_loss += (loss / n)
    return val_loss.item()


def rl_train(model: nn.Module, tokenizer: CharTokenizer, token_trie: TokenTrie,
             config: dict, optimizer: Optional[optim.Optimizer] = None, use_curiosity: bool = True):
    """Fine-tuning the model on a word generation task using Reinforcement Learning"""
    rl_cfg = config["rl_cfg"]
    max_len = config["model_cfg"]["max_word_len"]
    self_critic, batch_size, entr_penalty, n_eval_batches = \
        (rl_cfg[k] for k in ("self_critic", "batch_size", "entr_penalty", "n_eval_batches"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward = WordReward(token_trie, rl_cfg["status_reward_mapping"])
    if use_curiosity:
        curiosity_reward = CuriosityRewardTransformer(tokenizer.n_tokens, lr=1e-2, temperature=(10., 5., 1.), lr_t=1e-3)
        # calibrate curiosity
        with torch.no_grad():
            batch = model.generate_sample(100)
        curiosity_reward.calibrate_scale(batch[0].cpu(), rl_cfg["initial_curiosity"])
        reward = WeightedSumRewards(reward, curiosity_reward, 0.)
        coeff_upd_fun = lambda step: min(1., max(0., 0.01*(step - 10)))
    reward = QValueAggregator(reward_module=reward, max_len=max_len)
    model = model.to(device)
    reward = reward.to(device)
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=rl_cfg["lr"], weight_decay=1e-3)
    for step in range(rl_cfg["steps"]):
        if use_curiosity:
            reward.update_weight(coeff_upd_fun(step))
        mean_reward, seq = pg_step(model, reward, optimizer, batch_size, self_critic, entr_penalty)
        # metrics = calculate_rl_metrics(seq, token_trie)
        print(f"Step {step}: Mean reward {mean_reward}")
        do_every_k_step(step, 10, eval_sequence, seq, tokenizer, n=20)
        do_every_k_step(step, 10, count_generated_word_types, model, batch_size, n_eval_batches, token_trie)


def pg_step(model, reward, optimizer, batch_size: int, self_critic: bool = True, entr_penalty: float = 0.0):
    """A single Policy Gradient (aka REINFORCE) step"""
    model.train()
    optimizer.zero_grad()
    seq, log_probs, entropy = model.generate_sample(batch_size)
    advantage = reward(seq)
    mean_rew = advantage[:, 0].mean().item()
    if self_critic:
        seq_baseline = model.generate_argmax(batch_size)
        advantage -= reward(seq_baseline)
    entropy[seq == 0] = 0.0
    # entropy values are positive
    pg_loss = -(advantage * log_probs + entr_penalty*entropy).mean()
    pg_loss.backward()
    optimizer.step()
    return mean_rew, seq


def ppo_step(model, reward, optimizer, batch_size: int, self_critic: bool=True, entr_penalty: float = 0.0):
    """A single Proximity Policy Optimization step"""
    pass


def calculate_rl_metrics(seq, token_trie):
    pass


@torch.no_grad()
def count_generated_word_types(model, batch_size, n_batches, token_trie):
    model.eval()
    seq_list = []
    for _ in range(n_batches):
        seq, _, _ = model.generate_sample(batch_size)
        seq_list.append(seq.cpu())
    seq = torch.cat(seq_list, dim=0)
    del seq_list
    seq = seq.unique(sorted=False, dim=0)
    n_dups = batch_size * n_batches - seq.shape[0]
    # the first column of 'results' tensor is a word (prefix) type, the second one is the full word indicator
    results: torch.Tensor = token_trie.check(seq)
    results = results[results[:, 1] == 1]
    results = results[:, 0].type(torch.uint8)
    count = collections.Counter(results.tolist())
    count = {k: count[v] for k, v in zip(["word", "test_word", "train_word"], [1, 2, 3])}
    print(f" -- got {count} out of {batch_size*n_batches} generated word-likes with {n_dups} duplicates")


def load_config(fname="config.json") -> dict:
    with open(fname, "r") as f:
        config = json.load(f)
    return config


def do_every_k_step(step: int, k: int, fun: Callable, *args, **kwargs):
    if (step % k) == 0:
        fun(*args, **kwargs)


@torch.no_grad()
def eval_model(model, tokenizer, n_samples=50):
    model.eval()
    seq, _, _ = model.generate_sample(n_samples)
    eval_sequence(seq, tokenizer)


def eval_sequence(seq: torch.Tensor, tokenizer, n: Optional[int] = None):
    seq = seq.cpu().tolist()
    if n is not None:
        seq = seq[:n]
    seq = tokenizer.decode(seq)
    seq = [extract_decoded(s) for s in seq]
    print(seq)


def extract_decoded(string):
    result = re.search('<(.*)>', string)
    string = "" if result is None else result.group(1)
    return string


def write_to_summary(writer, metrics: dict, global_step: int):
    for k, v in metrics.items():
        writer.add_scalar(k, v.item(), global_step=global_step)


def main(config: dict):
    model, tokenizer, token_word_checker, str_word_checker = pretrain(config)
    # fine-tune only the last transformer layer and the model's 'head'
    model.freeze_n_layers(config["model_cfg"]["num_layers"] - 1)
    rl_train(model, tokenizer, token_word_checker, config, optimizer=None, use_curiosity=True)


def param_range(model):
    maxx = float("-inf")
    minn = float("inf")
    for par in model.parameters():
        maxx = max(maxx, par.max().item())
        minn = min(minn, par.min().item())
    return minn, maxx


if __name__ == "__main__":
    cfg = load_config("config_2.json")
    main(cfg)

    print("woah!")
