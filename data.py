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

    df["train/test"] = np.zeros(len(df), dtype=np.uint8)
    df.loc[targ_idxs_train, "train/test"] = 1
    df.loc[targ_idxs_test, "train/test"] = 2


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


class WordChecker:
    """A class that classifies an input word as a word, a non-word or a word from the target part of speech."""
    def __init__(self, dictionary: pd.DataFrame, target_pos: str):
        self._word_series = dictionary["word"]
        self._pos = dictionary["pos"].to_numpy()
        self._target_train_test = dictionary["train/test"].to_numpy()
        self._target_pos = target_pos
        self._n = len(dictionary)

    @handle_str_batches
    def check(self, word) -> WordStatus:
        # using built-in pandas binary search via searchsorted
        idx = self._word_series.searchsorted(word)
        word_q = self._word_series.iloc[idx]
        if idx == self._n or word_q != word:
            return WordStatus.NONWORD
        # words can belong to multiple parts of speech, so we have to check next entries
        while word_q == word:
            if self._target_train_test[idx] == 1:
                return WordStatus.TARGWORD_TRAIN
            if self._target_train_test[idx] == 2:
                return WordStatus.TARGWORD_TEST
            idx += 1
            word_q = self._word_series.iloc[idx]
        return WordStatus.WORD


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


def create_words_data(path: Pathlike, rl_target: str = "v. t.", train_test_proportion=(1, 2)):
    # rl_target is the set of words defined by 'pos' column (part of speech) that is used for RL fine-tuning
    dictionary = load_dictionary(path, lower=True, filter_missing=False)
    _deal_with_duplicates(dictionary, rl_target)
    _add_train_test_split_column(dictionary, rl_target, train_test_proportion)
    alphabet = get_alphabet(dictionary["word"])
    is_word = WordChecker(dictionary, rl_target)
    pretrain_words_series = dictionary.loc[dictionary["train/test"] < 2, "word"]
    return pretrain_words_series, is_word, alphabet


class WordsDataset(TensorDataset):
    def __init__(self, word_series: pd.Series, tokenizer: CharTokenizer):
        n = word_series.map(len).max()
        # adding 'begin sequence', 'end sequence' and 'padding' characters
        data = word_series.map(lambda string: "".join(["<", string, ">", "*"*(n - len(string))]))
        data = tokenizer.encode(data)   # pd.Series of tuples of ints
        data = torch.tensor(data, dtype=torch.uint8)
        inputs = data[:, :-1]
        targets = data[:, 1:]
        super().__init__(inputs, targets)


def get_data(data_cfg: Dict) -> Tuple[DataLoader, CharTokenizer, WordChecker]:
    """Outputs pretrain dataloaer, tokenizer (for decoding), word checker (for rl finetuning)"""
    print("Dataset preparation ...", end="")
    pretrain_words_series, is_word, alphabet = create_words_data(data_cfg["path"], data_cfg["target_pos"],
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
    return pretrain_dataloader, tokenizer, is_word


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
