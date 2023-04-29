import json
import os.path
import re
from collections import defaultdict
from typing import Dict, List, Optional, Union

from tqdm import tqdm

from tokenizers.base_tokenizer import BaseTokenizer


class SimpleTokenizer(BaseTokenizer):
    def __init__(
        self,
        path: str = "",
        max_words: int = 10000,
    ) -> None:
        super().__init__()
        self._word2id: Optional[Dict[str, int]] = None
        self._id2word: Optional[Dict[int, str]] = None
        self._max_words = max_words
        self.load(path)

    def encode(self, text: str) -> List[int]:
        words = [
            self.preprocessor(word)
            for word in self.preprocessor(text).split(" ")
            if word
        ]
        tokens = [self._word2id.get(word, self.unk_token) for word in words if word]
        return tokens

    def encode_word(self, word: str) -> int:
        return self._word2id.get(self.preprocessor(word), self.unk_token)

    def decode(self, tokens: List[int]) -> str:
        tokens = [int(token) for token in tokens]
        words = [self.decode_token(token) for token in tokens]
        return " ".join(words)

    def decode_token(self, token: int) -> str:
        return self._id2word.get(int(token), self.unk_token_name)

    def fit(
        self,
        data: Union[str, List[str]],
        verbose: bool = True,
        max_words: Optional[int] = None,
    ) -> None:
        self._max_words = max_words or self._max_words
        word2cnt = defaultdict(lambda: 0)
        word_list = self._extract_words(data)
        with tqdm(total=len(word_list)) as pbar:
            for idx, word in enumerate(word_list):
                if len(word) == 0:
                    continue
                word2cnt[word] += 1
                if idx % 10000 == 0:
                    pbar.set_description(f"Processed {idx + 1}/{len(word_list)}")
                pbar.update(1)
        loaded_cnt = len(word2cnt)
        if verbose:
            print(
                f"Loaded {loaded_cnt} unique words from {len(word_list)} corpus"
                f" ({100. * loaded_cnt / len(word_list)} %)"
            )
        words_limit = self._max_words - len(self.spec_tokens)
        popular_words = sorted(word2cnt.items(), reverse=True, key=lambda x: x[1])[
            :words_limit
        ]
        if verbose:
            print(f"20 most popular words:\n{popular_words[:20]}")

        self._word2id = {item.name: item.value for item in self.spec_tokens}
        for word, _ in popular_words:
            self._word2id[word] = len(self._word2id)
        assert len(self._word2id) <= self._max_words
        self._id2word = {v: k for k, v in self._word2id.items()}

    def _extract_words(self, data: Union[str, List[str]]) -> List[str]:
        words = []
        if isinstance(data, list):
            for line in data:
                assert isinstance(line, str)
                words.extend([self.preprocessor(x) for x in line.split(" ")])
        else:
            assert isinstance(data, str)
            words.extend([self.preprocessor(x) for x in data.split(" ")])
        return words

    def __len__(self) -> int:
        return len(self._word2id)

    @staticmethod
    def preprocessor(word: str) -> str:
        return re.sub("[^A-Za-zА-Яа-я\s]+", "", word).lower().strip()

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self._word2id, f)

    def load(self, path: str) -> None:
        if os.path.isfile(path):
            with open(path, "r") as f:
                self._word2id = json.load(f)
            self._id2word = {v: k for k, v in self._word2id.items()}
            self._max_words = len(self._word2id)
