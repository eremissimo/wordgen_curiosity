import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from enum import Enum
from typing import List, Union, Tuple, Dict, Iterable, Callable, Type, Optional
from functools import wraps, partial


Pathlike = Union[Path, str]


class WordStatus(Enum):
    NONWORD = 0
    WORD = 1
    TARGWORD_TRAIN = 2
    TARGWORD_TEST = 3

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
    def wrapper(self, input):
        if single_instance_check_fn(input):
            # single
            return fcn_single(self, input)
        # batched
        batch_type = type(input)
        return batch_type(fcn_single(self, w) for w in input)
    return wrapper


handle_str_batches = partial(_handle_batches, single_instance_check_fn=lambda x: isinstance(x, str))
handle_int_batches = partial(_handle_batches, single_instance_check_fn=lambda x: isinstance(next(iter(x)), int))
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

    def __init__(self, value: Optional[WordStatus] = None):
        self.children: Dict[str, CharTrieNode] = {}
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

    def reset(self):
        self.root = CharTrieNode()
        self.node_count = 0


def create_words_data(path: Pathlike, rl_target: str = "v. t.", train_test_proportion=(1, 2)):
    # rl_target is the set of words defined by 'pos' column (part of speech) that is used for RL fine-tuning
    dictionary = load_dictionary(path, lower=True, filter_missing=False)
    _deal_with_duplicates(dictionary, rl_target)
    _add_train_test_split_column(dictionary, rl_target, train_test_proportion)
    alphabet = get_alphabet(dictionary["word"])
    word_checker = CharTrie(modify_value_fn=max)
    word_checker.from_iterable(dictionary["word"], dictionary["train/test"])
    pretrain_words_series = dictionary.loc[dictionary["train/test"] < WordStatus.TARGWORD_TEST, "word"]
    return pretrain_words_series, word_checker, alphabet


class WordsDataset(TensorDataset):
    def __init__(self, word_series: pd.Series, tokenizer: CharTokenizer):
        n = word_series.map(len).max()
        # adding 'begin sequence', 'end sequence' and 'padding' characters
        data = word_series.map(lambda string: "".join(["<", string, ">", "*"*(n - len(string))]))
        data = tokenizer.encode(data)   # pd.Series of tuples of ints
        data = torch.tensor(data, dtype=torch.long)
        inputs = data[:, :-1]
        targets = data[:, 1:]
        super().__init__(inputs, targets)


def get_data(data_cfg: Dict) -> Tuple[DataLoader, CharTokenizer, CharTrie]:
    """Outputs pretrain dataloaer, tokenizer (for decoding), word checker (for rl finetuning)"""
    print("Dataset preparation ...", end="")
    pretrain_words_series, word_checker, alphabet = create_words_data(data_cfg["path"], data_cfg["target_pos"],
                                                                      data_cfg["train_test_proportion"])
    print(".", end="")
    tokenizer = CharTokenizer(alphabet)
    pretrain_dataset = WordsDataset(pretrain_words_series, tokenizer)
    print(". ", end="")
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=data_cfg["batch_size"], shuffle=True,
                                     num_workers=data_cfg["num_workers"], pin_memory=True, drop_last=False,
                                     pin_memory_device="cuda")
    print("Done!")
    print("Using alphabet: ", alphabet)
    return pretrain_dataloader, tokenizer, word_checker


if __name__ == "__main__":
    cfg = {"path": "dictionary.csv",
           "target_pos": "v. t.",
           "train_test_proportion": (1, 2),
           "batch_size": 4,
           "num_workers": 0}
    dataloader, _, checker = get_data(cfg)
    print(next(iter(dataloader)))
    print(checker.check(['aaaa', 'alienate', 'worm', 'sacrilegious', 'run', 'beat', 'write', 'lol', 'cheetah']))
    print("woah!")
