from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from tokenizers import BaseTokenizer
from utils import pad


class ChatDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: BaseTokenizer,
        max_words: int = 60,
        limit: int = -1,
    ):
        df = pd.read_csv(path)
        if limit > 0:
            idxs = np.arange(len(df))
            idxs = np.random.choice(idxs, limit, replace=limit > len(df))
            df = df.iloc[idxs].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.in_sentences = df["question"].values
        self.target_sentences = df["answer"].values
        enc_in_sentences_mask = np.array([
            0 < len(self.tokenizer.encode(val)) < self.max_words - 2
            for val in tqdm(self.in_sentences, desc="enc_in_sentences_mask")
        ])
        enc_target_sentences_mask = np.array([
            0 < len(self.tokenizer.encode(val)) < self.max_words - 2
            for val in tqdm(self.target_sentences, desc="enc_target_sentences_mask")
        ])
        mask = enc_in_sentences_mask & enc_target_sentences_mask
        self.in_sentences = self.in_sentences[mask]
        self.target_sentences = self.target_sentences[mask]
        print("Dataset size:", len(self))

    def __len__(self):
        return len(self.in_sentences)

    def __getitem__(self, index):
        in_sentence = self.in_sentences[index]
        target_sentence = self.target_sentences[index]
        enc_in_sentence = self.tokenizer.encode(in_sentence)
        enc_target_sentence = self.tokenizer.encode(target_sentence)
        encoder_input = torch.tensor(
            pad(enc_in_sentence, self.tokenizer, self.max_words, prepadding=True),
            dtype=torch.long
        )
        decoder_input = torch.tensor(
            pad(
                [self.tokenizer.start_token] + enc_target_sentence + [self.tokenizer.stop_token],
                self.tokenizer,
                self.max_words
            ),
            dtype=torch.long
        )
        decoder_output = torch.tensor(
            pad(
                enc_target_sentence + [self.tokenizer.stop_token],
                self.tokenizer,
                self.max_words
            ),
            dtype=torch.long
        )
        return {
            "in_sentence": in_sentence,
            "target_sentence": target_sentence,
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "decoder_output": decoder_output,
        }
