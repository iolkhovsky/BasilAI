from abc import ABC, abstractmethod
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional


class BaseSpecToken(IntEnum):
    PAD = 0
    UNK = 1
    START = 2
    STOP = 3


class BaseTokenizer(ABC):
    """
    BaseTokenizer class
    """

    def __init__(self, spec_tokens: Enum = None):
        self.spec_tokens = spec_tokens or BaseSpecToken

    @abstractmethod
    def encode(self, text: str) -> List[Any]:
        pass

    @abstractmethod
    def encode_word(self, word: str) -> int:
        pass

    @abstractmethod
    def decode(self, tokens: List[Any]) -> str:
        pass

    @abstractmethod
    def decode_token(self, token: int) -> str:
        pass

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    def start_token(self) -> int:
        return self.spec_tokens.START.value

    @property
    def stop_token(self) -> int:
        return self.spec_tokens.STOP.value

    @property
    def pad_token(self) -> int:
        return self.spec_tokens.PAD.value

    @property
    def unk_token(self) -> int:
        return self.spec_tokens.UNK.value

    @property
    def start_token_name(self) -> str:
        return self.spec_tokens.START.name

    @property
    def stop_token_name(self) -> str:
        return self.spec_tokens.STOP.name

    @property
    def pad_token_name(self) -> str:
        return self.spec_tokens.PAD.name

    @property
    def unk_token_name(self) -> str:
        return self.spec_tokens.UNK.name

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @classmethod
    def from_pretrained(cls, path: str):
        obj = cls()
        obj.load(path)
        return obj
