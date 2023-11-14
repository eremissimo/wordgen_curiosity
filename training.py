import json
import re
import collections
import torch
import torch.nn as nn
import torch.nn.functional as ff
import torch.optim as optim
from typing import Optional, Callable
from copy import deepcopy
from tqdm import tqdm

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
        do_every_k_step(epoch, 20, save_checkpoint, model, f"./checkpoints/ChTrans_2_{val_loss:.4f}.pt")
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
             config: dict, optimizer: Optional[optim.Optimizer] = None):
    """Fine-tuning the model on a word generation task using Reinforcement Learning"""
    rl_cfg = config["rl_cfg"]
    max_len = config["model_cfg"]["max_word_len"]
    self_critic, use_curiosity, batch_size, entr_penalty, n_eval_batches = \
        (rl_cfg[k] for k in ("self_critic", "use_curiosity", "batch_size", "entr_penalty", "n_eval_batches"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward = WordReward(token_trie, rl_cfg["status_reward_mapping"])
    if use_curiosity:
        curiosity_reward = CuriosityRewardTransformer(tokenizer.n_tokens, lr=1e-2, temperature=(4., 1., 1.), lr_t=1e-3)
        # calibrate curiosity
        batch = model.generate_sample(100)
        curiosity_reward.calibrate_scale(batch.cpu(), rl_cfg["initial_curiosity"])
        reward = WeightedSumRewards(reward, curiosity_reward, 0.)
        coeff_upd_fun = lambda step: min(1., max(0., 0.02*(step - 0)))
        # coeff_upd_fun = lambda step: 1
    reward = QValueAggregator(reward_module=reward, max_len=max_len)
    model = model.to(device)
    reward = reward.to(device)
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=rl_cfg["lr"], weight_decay=1e-3)
    for step in range(rl_cfg["steps"]):
        if use_curiosity:
            reward._reward.update_weight(coeff_upd_fun(step))
        mean_reward, seq = pg_step(model, reward, optimizer, batch_size, self_critic, entr_penalty)
        # metrics = calculate_rl_metrics(seq, token_trie)
        print(f"Step {step}: Mean reward {mean_reward}")
        do_every_k_step(step, 10, eval_sequence, seq, tokenizer, n=20)
        do_every_k_step(step, 10, count_generated_word_types, model, batch_size, n_eval_batches, token_trie)


def pg_step(model, reward, optimizer, batch_size: int, self_critic: bool = True, entr_penalty: float = 0.0):
    """A single Policy Gradient (aka REINFORCE) step"""
    model.train()
    optimizer.zero_grad()
    seq, log_probs, entropy = model.generate_sample_train(batch_size)
    seq, log_probs, entropy = refine_predictions(seq, log_probs, entropy)
    advantage = reward(seq)
    mean_rew = advantage[:, 0].mean().item()
    if self_critic:
        seq_baseline = model.generate_argmax(batch_size)
        seq_baseline = refine_predictions(seq_baseline)[0]
        advantage -= reward(seq_baseline, optim_step=False)
    entropy[seq == 0] = 0.0
    # entropy values are positive
    pg_loss = -(advantage * log_probs + entr_penalty*entropy).mean()
    pg_loss.backward()
    optimizer.step()
    return mean_rew, seq


def refine_predictions(*tensors):
    """Get rid of some cursed predictions: '<word-like>**<<asdf><**' -> '<word-like>**********' """
    end_token = 2    # :/
    filled_pad_mask = tensors[0] == end_token
    has_end, arg_first_end = filled_pad_mask.max(dim=1, keepdim=True)
    arg_first_end[~has_end] = filled_pad_mask.shape[1] - 1
    filled_pad_mask = (torch.arange(filled_pad_mask.shape[1],
                                    dtype=torch.short,
                                    device=filled_pad_mask.device).unsqueeze(0) > arg_first_end)
    for t in tensors:
        t[filled_pad_mask] = 0
    return tensors


def count_generated_word_types(model, batch_size, n_batches, token_trie):
    model.eval()
    seq_list = []
    for _ in range(n_batches):
        seq = model.generate_sample(batch_size)
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


def eval_sequence(seq: torch.Tensor, tokenizer, n: Optional[int] = None):
    seq = seq.cpu().tolist()
    if n is not None:
        seq = seq[:n]
    seq = tokenizer.decode(seq)
    seq = [extract_decoded(s) for s in seq]
    print(seq)


def extract_decoded(string):
    """ '<word-like>***********' -> 'word-like'  """
    result = re.search('<(.*)>', string)
    string = "" if result is None else result.group(1)
    return string


def main(config: dict):
    """The main training function. Pretrain on the set (Words \ Test_v.t._words),
    then fine-tune on the set (Train_v.t._words)"""
    model, tokenizer, token_word_checker, str_word_checker = pretrain(config)
    # fine-tune only the last transformer layer and the model's 'head'
    model.freeze_n_layers(config["model_cfg"]["num_layers"] - 1)
    rl_train(model, tokenizer, token_word_checker, config, optimizer=None)


def word_discovery(config: dict, batch_size: int, n_batches: int, use_checkpoint: Optional[str] = None):
    """Sample the trained model loaded from the 'use_checkpoint' and discover unseen v.t. words
    (i.e. words from the test set)"""
    config = deepcopy(config)
    config["pretrain_cfg"]["epochs"] = 0        # skip pretraining, only load data
    if use_checkpoint is not None:
        config["pretrain_cfg"]["checkpoint"] = use_checkpoint
    model, tokenizer, token_word_checker, _ = pretrain(config)
    n_verbs_overall = token_word_checker.summary[2]
    model.freeze()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    trans_verbs = set()
    test_word_type = 2
    for _ in tqdm(range(n_batches)):
        seq = model.generate_sample(batch_size)
        word_type: torch.Tensor = token_word_checker.check(seq.cpu())
        # taking only full words from test data
        seq = seq[(word_type[:, 0] == test_word_type) & word_type[:, 1].bool()]
        if seq.shape[0] == 0:
            continue
        words_str = tokenizer.decode(seq.tolist())
        trans_verbs.update(map(extract_decoded, words_str))
    print("WordDiscovery: ")
    print(" - model checkpoint: ", config["pretrain_cfg"]["checkpoint"])
    print(f" - found verbs: {len(trans_verbs)}/{n_verbs_overall} from {batch_size*n_batches} samples")
    return trans_verbs


if __name__ == "__main__":
    cfg = load_config("config_2.json")
    main(cfg)
    # verbs = word_discovery(cfg, 2048, 100, use_checkpoint="checkpoints/ChTrans_382_entr.pt")
    # print(verbs)

    print("woah!")
