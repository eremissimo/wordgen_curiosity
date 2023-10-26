import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
from pathlib import Path
from enum import Enum
from typing import List, Union, Tuple, Dict, Iterable, Callable, Optional
from functools import wraps, partial


Pathlike = Union[Path, str]


class WordStatus(Enum):
    NONWORD = 0
    WORD = 1
    TARGWORD_TEST = 2
    TARGWORD_TRAIN = 3

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

    def __str__(self):
        return self.name


def load_dictionary(path: Pathlike, lower=True, filter_missing=False) -> pd.DataFrame:
    # keep_default_na = False because we don't want to interpret valid words like 'None', 'NaN' etc. as missing values
    df = pd.read_csv(path, header=None, names=("idx", "word", "pos"),
                     index_col="idx", dtype={"idx": int, "word": str, "pos": str}, keep_default_na=False)
    if lower:
        df["word"] = df["word"].map(str.lower)
    if filter_missing:
        df = df[df["pos"].map(len) > 0]
    return df


def _deal_with_duplicates(df: pd.DataFrame, target_pos: str):
    # this is the easiest way to drop duplicate words (we don't care too much about other p.o.s. besides target_pos)
    # leaving the list of target_pos words intact for target_pos="v. t."
    # TODO: for other target_pos there must be more involved procedure, maybe I'll do it but not now
    assert target_pos == "v. t.", f"target_pos == {target_pos} is not yet supported"
    df.drop_duplicates("word", keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)


def _add_train_test_split_column(df: pd.DataFrame, target_pos: str, train_test_proportion: List | Tuple):
    # assuming there are no duplicates in the dictionary
    targ_idxs_all = df.index[df["pos"] == target_pos]
    n_targ = len(targ_idxs_all)
    n_targ_train = int(n_targ * train_test_proportion[0] / sum(train_test_proportion))
    shuffle_idxs = np.random.permutation(n_targ)
    targ_idxs_train = targ_idxs_all[shuffle_idxs[:n_targ_train]]
    targ_idxs_test = targ_idxs_all[shuffle_idxs[n_targ_train:]]

    df["train/test"] = WordStatus.WORD
    df.loc[targ_idxs_train, "train/test"] = WordStatus.TARGWORD_TRAIN
    df.loc[targ_idxs_test, "train/test"] = WordStatus.TARGWORD_TEST


def get_alphabet(words: pd.Series) -> str:
    chars = set()
    for i, word in enumerate(words):
        chars.update(word)
    alphabet = "".join(sorted(chars))
    return alphabet


def _handle_batches(fcn_single: Callable, single_instance_check_fn: Callable):
    """A template for decorators that handle batched input."""
    @wraps(fcn_single)
    def wrapper(self, inpt):
        if single_instance_check_fn(inpt):
            # single
            return fcn_single(self, inpt)
        # batched
        batch_type = type(inpt)
        return batch_type([fcn_single(self, w) for w in inpt])
    return wrapper


handle_str_batches = partial(_handle_batches, single_instance_check_fn=lambda x: isinstance(x, str))
handle_int_batches = partial(_handle_batches, single_instance_check_fn=lambda x: isinstance(next(iter(x)), int))
handle_array_batches = partial(_handle_batches, single_instance_check_fn=lambda x: x.ndim == 1)
# handle_int_batches = partial(_handle_batches, single_instance_check_fn=lambda x: isinstance(x[0], int))


class CharTokenizer:
    def __init__(self, alphabet: str, convert_fn: Optional[Callable] = None):
        # additional characters:
        # * padding
        # < start sequence (word)
        # > end sequence (word)
        assert not set(alphabet).intersection("*<>")
        self.alphabet = "".join(["*<>", alphabet])
        self.n_tokens = len(self.alphabet)
        self._encoder = {char: i for i, char in enumerate(self.alphabet)}
        self._decoder = self.alphabet
        self._convert_fn = convert_fn

    @handle_str_batches
    def encode(self, word: str) -> Iterable[int]:
        tokens = tuple(self._encoder[char] for char in word)
        if self._convert_fn is not None:
            tokens = self._convert_fn(tokens)
        return tokens

    @handle_int_batches
    def decode(self, idxs: Iterable[int]) -> str:
        return "".join(self._decoder[i] for i in idxs)


class CharTrieNode:
    __slots__ = ["children", "is_full_word", "value"]

    def __init__(self, value=None):
        self.children = {}
        self.is_full_word = False
        self.value = value


class CharTrie:
    """A word and prefixes checker. This is a character level trie implemented via nested dicts essentially.
    This data structure will be used later in RL fine-tuning to produce rewards for generated prefixes and full words.

    CharTrie.check is meant to identify if a given prefix is a prefix of a non-word, word, target train or
    test p.o.s. word. Combined with modify_value_fn = max it sets a priority for types, e.g. ['b', 'bu', 'buy'] are
    considered as target word prefixes tor target p.o.s = 'v. t.'. """
    def __init__(self, modify_value_fn=max):
        self.root = CharTrieNode()
        self.node_count = 0
        self._modify_value_fn = modify_value_fn     # this takes the current node value, other value and produces a
                                                    # new value for the current node

    def insert(self, word: str, value: WordStatus):
        current_node = self.root
        for char in word:
            if char not in current_node.children:
                self.node_count += 1
                current_node.children[char] = CharTrieNode(value)
            elif self._modify_value_fn is not None:
                current_node.value = value if current_node.value is None \
                    else self._modify_value_fn(current_node.value, value)
            current_node = current_node.children[char]
        current_node.is_full_word = True

    def from_iterable(self, words: Iterable[str], values: Iterable[WordStatus]):
        """Building the trie from iterables"""
        for word, value in zip(words, values):
            self.insert(word, value)

    @handle_str_batches
    def check(self, word) -> Tuple[WordStatus, bool]:
        """The main prefix search function"""
        current_node = self.root
        for char in word:
            if char not in current_node.children:
                return WordStatus.NONWORD, False
            current_node = current_node.children[char]
        return current_node.value, current_node.is_full_word

    @handle_str_batches
    def word_values(self, word) -> List[WordStatus]:
        current_node = self.root
        out = [WordStatus.NONWORD]*len(word)
        for i, char in enumerate(word):
            if char not in current_node.children:
                return out
            current_node = current_node.children[char]
            out[i] = current_node.value
        return out

    def reset(self):
        self.root = CharTrieNode()
        self.node_count = 0


class TokenTrie:
    def __init__(self, modify_value_fn=max, padding_idx=0, output_pad_idx=-1):
        self.root = CharTrieNode()
        self.node_count = 0
        self.padding_idx = padding_idx
        self.out_pad_idx = output_pad_idx
        self._modify_value_fn = modify_value_fn  # this takes the current node value, other value and produces a
        # new value for the current node

    def insert(self, tokenized_word: Iterable[int], value: int):
        # assuming tokenized_word is consist of the chain of token idxs followed by some amount of padding idxs
        if next(iter(tokenized_word)) == self.padding_idx:
            return
        current_node = self.root
        for idx in tokenized_word:
            if idx == self.padding_idx:
                break
            if idx not in current_node.children:
                self.node_count += 1
                current_node.children[idx] = CharTrieNode(value)
            elif self._modify_value_fn is not None:
                current_node.value = value if current_node.value is None \
                    else self._modify_value_fn(current_node.value, value)
            current_node = current_node.children[idx]
        current_node.is_full_word = True

    def from_iterable(self, tokenized_words: Iterable[Iterable[int]], values: Iterable[int]):
        """Building the trie from iterables"""
        for word, value in zip(tokenized_words, values):
            self.insert(word, value)

    @handle_array_batches
    def check(self, tokenized_word) -> Tuple[int, bool]:
        """Prefix search method"""
        if tokenized_word[0] == self.padding_idx:
            return 0, False
        current_node = self.root
        for idx in tokenized_word:
            idx = int(idx)
            if idx == self.padding_idx:
                break
            if idx not in current_node.children:
                return 0, False
            current_node = current_node.children[idx]
        return current_node.value, current_node.is_full_word

    def word_values(self, tokenized_words: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # *1 part is for inter-compatibility between torch.Tensor and np.ndarray
        out = torch.where(tokenized_words == self.padding_idx, self.out_pad_idx, 0)
        is_full_word = torch.zeros((tokenized_words.shape[0],), dtype=torch.bool)
        for i in range(tokenized_words.shape[0]):
            current_node = self.root
            nonword = False
            for j in range(tokenized_words.shape[1]):
                idx = int(tokenized_words[i, j])
                if idx == self.padding_idx:
                    break
                if idx not in current_node.children:
                    nonword = True
                    break
                current_node = current_node.children[idx]
                out[i, j] = current_node.value
            if not nonword:
                is_full_word[i] = current_node.is_full_word
        return out, is_full_word

    def reset(self):
        self.root = CharTrieNode()
        self.node_count = 0


def get_data(data_cfg: Dict) -> Tuple[DataLoader, DataLoader, CharTokenizer, TokenTrie, CharTrie]:
    """Outputs pretrain dataloaer, tokenizer (for decoding), word checker (for rl finetuning)"""
    print("Dataset preparation ...", end="")

    dictionary = load_dictionary(data_cfg["path"], lower=True, filter_missing=False)
    _deal_with_duplicates(dictionary, data_cfg["target_pos"])
    _add_train_test_split_column(dictionary, data_cfg["target_pos"], data_cfg["train_test_proportion"])
    alphabet = get_alphabet(dictionary["word"])
    tokenizer = CharTokenizer(alphabet)
    print(".", end="")
    # this word checker is built over chars and is being used at evaluation step (if any)
    eval_word_checker = CharTrie(modify_value_fn=max)
    eval_word_checker.from_iterable(dictionary["word"], dictionary["train/test"])
    print(".", end="")
    # adding 'begin sequence', 'end sequence' and 'padding' characters
    n = dictionary["word"].map(len).max()
    dictionary["word"] = dictionary["word"].map(lambda string: "".join(["<", string, ">", "*"*(n - len(string))]))
    dictionary["word"] = tokenizer.encode(dictionary["word"])  # pd.Series of tuples of ints
    print(".", end="")
    # this word checker is built over tokens (idxs) and is used to create rewards at rl training step
    rl_reward_word_checker = TokenTrie(modify_value_fn=max, padding_idx=0)
    rl_reward_word_checker.from_iterable(dictionary["word"], dictionary["train/test"].map(lambda ws: ws.value))
    print(".", end="")
    # pretrain dataset
    pretrain_dataset = dictionary.loc[dictionary["train/test"] != WordStatus.TARGWORD_TEST, "word"]
    pretrain_dataset = torch.tensor(pretrain_dataset, dtype=torch.long)
    pretrain_dataset = TensorDataset(pretrain_dataset[:, :-1], pretrain_dataset[:, 1:])
    pretrain_tr_data, pretrain_val_data = random_split(pretrain_dataset, lengths=[0.9, 0.1])
    print(". ", end="")
    # dataloaders
    pretrain_dataloader = DataLoader(pretrain_tr_data, batch_size=data_cfg["batch_size"], shuffle=True,
                                     num_workers=data_cfg["num_workers"], pin_memory=True, drop_last=False,
                                     pin_memory_device="cuda")
    pretrain_val_dataloader = DataLoader(pretrain_val_data, batch_size=data_cfg["batch_size"], shuffle=True,
                                         num_workers=data_cfg["num_workers"], pin_memory=True, drop_last=False,
                                         pin_memory_device="cuda")
    print("Done!")
    print("Using alphabet: ", alphabet)
    return pretrain_dataloader, pretrain_val_dataloader, tokenizer, rl_reward_word_checker, eval_word_checker


if __name__ == "__main__":
    cfg = {"path": "dictionary.csv",
           "target_pos": "v. t.",
           "train_test_proportion": (1, 2),
           "batch_size": 4,
           "num_workers": 0}
    dataloader, _, tok, tok_checker, checker = get_data(cfg)
    inp, targ = next(iter(dataloader))
    print(checker.check(['a', 'aa', 'aaa', 'aaaa', 'alienate', 'worm',
                         'sacrilegious', 'run', 'beat', 'write', 'lol', 'cheetah']))
    print(checker.word_values("wednesdayyy"))
    vals, is_full = tok_checker.word_values(inp)
    print(inp)
    print(vals)
    print(is_full)

    print("woah!")
