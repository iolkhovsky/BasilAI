from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from tokenizers import BaseTokenizer
from utils import pad


class ChatDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: BaseTokenizer,
        max_words: int = 60,
        limit: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self._max_words = max_words
        self._df = pd.read_csv(path)
        if limit:
            self._df = self._df.head(limit)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        in_sentence = self._df["question"][index]
        encoder_input = self.tokenizer.encode(in_sentence)[: self._max_words]
        encoder_input = pad(
            encoder_input, self.tokenizer, self._max_words, prepadding=True
        )

        target_sentence = self._df["answer"][index]
        decoder_output = self.tokenizer.encode(target_sentence) + [
            self.tokenizer.stop_token
        ]
        decoder_output = decoder_output[: self._max_words]
        decoder_input = [self.tokenizer.start_token] + decoder_output
        decoder_input = decoder_input[: self._max_words]
        decoder_output = pad(decoder_output, self.tokenizer, self._max_words)
        decoder_input = pad(decoder_input, self.tokenizer, self._max_words)

        assert len(encoder_input) == self._max_words
        assert len(decoder_input) == self._max_words
        assert len(decoder_output) == self._max_words

        return {
            "in_sentence": in_sentence,
            "target_sentence": target_sentence,
            "encoder_input": torch.tensor(encoder_input, dtype=torch.long),
            "decoder_input": torch.tensor(decoder_input, dtype=torch.long),
            "decoder_output": torch.tensor(decoder_output, dtype=torch.long),
        }
