import json
import torch
import torch.nn as nn
import torch.nn.functional as ff
import torch.optim as optim
from typing import List, Optional

from data import get_data, TokenTrie, CharTokenizer
from model import CharTransformer, save_checkpoint, load_checkpoint, SumRewards, WordReward, CuriosityRewardTransformer, \
    DualAscentPGLoss, RewardPGLoss


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
    self_critic, batch_size, entr_penalty = rl_cfg["self_critic"], rl_cfg["batch_size"], rl_cfg["entr_penalty"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward = WordReward(token_trie, rl_cfg["status_reward_mapping"])
    if use_curiosity:
        curiosity_reward = CuriosityRewardTransformer(tokenizer.n_tokens)
        # calibrate curiosity
        with torch.no_grad():
            batch = model.generate_sample(100)
        curiosity_reward.calibrate_scale(batch[0], rl_cfg["initial_curiosity"])
        reward = SumRewards(reward, curiosity_reward)
    model = model.to(device)
    reward = reward.to(device)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=rl_cfg["lr"])
    for step in range(rl_cfg["steps"]):
        mean_reward, seq = pg_step(model, reward, optimizer, batch_size, self_critic, entr_penalty)
        # metrics = calculate_rl_metrics(seq, token_trie)
        print(f"Step {step}: Mean reward {float(mean_reward)}")


def pg_step(model, reward, optimizer, batch_size: int, self_critic: bool =True, entr_penalty: float = 0.0):
    """A single Policy Gradient (aka REINFORCE) step"""
    model.train()
    optimizer.zero_grad()
    seq, log_probs, entropy = model.generate_sample(batch_size)
    advantage = reward(seq)
    mean_rew = advantage.mean().item()
    if self_critic:
        seq_baseline = model.generate_argmax(batch_size)
        advantage -= reward(seq_baseline)
    entropy[seq == 0] = 0.0
    # entropy values are positive
    pg_loss = -(advantage * log_probs + entr_penalty*entropy).mean()
    pg_loss.backward()
    optimizer.step()
    return mean_rew, seq


def pg_step_w_loss(model, loss_fn, optimizer, batch_size: int, self_critic: bool =True):
    model.train()
    optimizer.zero_grad()
    seq, log_probs, entropy = model.generate_sample(batch_size)
    seq_baseline = model.generate_argmax(batch_size) if self_critic else None
    entropy[seq == 0] = 0.0
    pg_loss, mean_rew = loss_fn(seq, log_probs, entropy, states_baseline=seq_baseline)
    pg_loss.backward()
    optimizer.step()
    return mean_rew, seq


def rl_train_w_loss(model: nn.Module, tokenizer: CharTokenizer, token_trie: TokenTrie,
                    config: dict, optimizer: Optional[optim.Optimizer] = None):
    """Fine-tuning the model on a word generation task using Reinforcement Learning"""
    rl_cfg = config["rl_cfg"]
    self_critic, batch_size, entr_penalty = rl_cfg["self_critic"], rl_cfg["batch_size"], rl_cfg["entr_penalty"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward = WordReward(token_trie, rl_cfg["status_reward_mapping"])
    curiosity_reward = CuriosityRewardTransformer(tokenizer.n_tokens)
    # calibrate curiosity
    with torch.no_grad():
        batch = model.generate_sample(100)
    curiosity_reward.calibrate_scale(batch[0].cpu(), rl_cfg["initial_curiosity"])
    loss_fn_dapg = DualAscentPGLoss(curiosity_reward, reward, target_reward=5.,
                               entr_penalty=entr_penalty, initial_lambda=2.).to(device)
    loss_fn_rpg = RewardPGLoss(reward, entr_penalty).to(device)
    model = model.to(device)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=rl_cfg["lr"])
    for step in range(rl_cfg["steps"]):

        loss_fn = loss_fn_rpg if step < 20 else loss_fn_dapg
        mean_reward, seq = pg_step_w_loss(model, loss_fn, optimizer, batch_size, self_critic)
        # metrics = calculate_rl_metrics(seq, token_trie)
        print(f"Step {step}: Mean reward {float(mean_reward)}")


def ppo_step(model, reward, optimizer, batch_size: int, self_critic: bool =True, entr_penalty: float = 0.0):
    """A single Proximity Policy Optimization step"""
    pass


def calculate_rl_metrics(seq, token_trie):
    pass


def load_config(fname="config.json") -> dict:
    with open(fname, "r") as f:
        config = json.load(f)
    return config


@torch.no_grad()
def evaluate(model, tokenizer, n_samples=50) -> List[str]:
    model.eval()
    seq, _, _ = model.generate_sample(n_samples)
    return eval_sequence(seq, tokenizer)


def eval_sequence(seq: torch.Tensor, tokenizer) -> List[str]:
    seq = seq.cpu().tolist()
    seq = tokenizer.decode(seq)
    return seq


def write_to_summary(writer, metrics: dict, global_step: int):
    for k, v in metrics.items():
        writer.add_scalar(k, v.item(), global_step=global_step)


def main(config: dict):
    model, tokenizer, token_word_checker, str_word_checker = pretrain(config)
    # fine-tune only the last transformer layer and the model's 'head'
    model.freeze_n_layers(config["model_cfg"]["num_layers"] - 1)
    rl_train_w_loss(model, tokenizer, token_word_checker, config, optimizer=None)


if __name__ == "__main__":
    cfg = load_config("config_2.json")
    main(cfg)

    print("woah!")
