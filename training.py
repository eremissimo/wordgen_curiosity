import json
import torch
import torch.nn as nn
import torch.nn.functional as ff
import torch.optim as optim

from data import get_data
from model import CharTransformer, save_checkpoint, load_checkpoint, SumRewards, WordReward, CuriosityRewardTransformer


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
    if pretrain_cfg["checkpoint"] is not None:
        pass
    for epoch in range(pretrain_cfg["epochs"]):
        train_loss = train_epoch(model, optimizer, pretrain_dataloader, device)
        val_loss = val_epoch(model, pretrain_val_dataloader, device)
        scheduler.step()
        print(f"epoch {epoch}: train: {train_loss} | val: {val_loss}")
        if (epoch + 1) % 35 == 0:
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


def rl_train(model, tokenizer, token_trie, config, optimizer=None):
    """"""
    rl_cfg = config["rl_cfg"]
    self_critic, batch_size = rl_cfg["self_critic"], rl_cfg["batch_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    curiosity_reward = CuriosityRewardTransformer(tokenizer.n_tokens)
    word_reward = WordReward(token_trie, rl_cfg["status_reward_mapping"])
    reward = SumRewards([word_reward, curiosity_reward]).to(device)
    model = model.to(device)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=rl_cfg["lr"])
    # calibrate curiosity
    batch = model.generate_sample(100)
    curiosity_reward.calibrate_scale(batch, rl_cfg["initial_curiosity"])      # TODO: check if this works properly
    for step in range(rl_cfg["steps"]):
        mean_reward, seq = pg_step(model, reward, optimizer, batch_size, self_critic)
        # metrics = calculate_rl_metrics(seq, token_trie)
        epoch, substep = divmod(step, 100)
        if substep == 0:
            print(f"Epoch {epoch}: Mean reward {mean_reward}")


def pg_step(model, reward, optimizer, batch_size, self_critic=True):
    """A single Vanilla Policy Gradient (aka REINFORCE) step"""
    optimizer.zero_grad()
    seq, log_probs = model.generate_sample(batch_size)
    advantage = reward(seq)
    mean_rew = advantage.mean()
    if self_critic:
        with torch.no_grad():
            seq_baseline = model.generate_argmax(batch_size)
        advantage -= reward(seq_baseline)
    pg_loss = (advantage * log_probs).mean()
    pg_loss.backward()
    optimizer.step()
    return mean_rew, seq


def ppo_step(model, reward, optimizer, batch_size, self_critic=True):
    """A single Proximity Policy Optimization step"""
    pass


def calculate_rl_metrics(seq, token_trie):
    pass


def load_config(fname="config.json") -> dict:
    with open(fname, "r") as f:
        cfg = json.load(f)
    return cfg


def write_to_summary(writer, metrics: dict, global_step: int):
    for k, v in metrics.items():
        writer.add_scalar(k, v.item(), global_step=global_step)


def main(config: dict):
    model, tokenizer, token_word_checker, str_word_checker = pretrain(config)
    rl_train(model, tokenizer, token_word_checker, config, None)


if __name__ == "__main__":
    cfg = load_config("config_2.json")
    main(cfg)

    print("woah!")
