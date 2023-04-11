import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils import pad


class ChatDataset(Dataset):
    def __init__(
        self,
        path,
        tokenizer,
        fit_tokenizer=False,
        max_words=60,
        limit=None
    ):
        self.tokenizer = tokenizer
        self._max_words = max_words
        self._df = pd.read_csv(path)
        if limit:
            self._df = self._df.head(limit)
       
        if fit_tokenizer:
            self.tokenizer.fit(list(self._df['answer'] + self._df['question']))

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):       
        in_sentence = self._df['question'][index]
        encoder_input = self.tokenizer.encode_line(in_sentence)[:self._max_words]
        encoder_input = pad(encoder_input, self.tokenizer, self._max_words, prepadding=True)

        target_sentence = self._df['answer'][index]
        decoder_output = self.tokenizer.encode_line(target_sentence) + [self.tokenizer.stop_token]
        decoder_output = decoder_output[:self._max_words]
        decoder_input = [self.tokenizer.start_token] + decoder_output
        decoder_input = decoder_input[:self._max_words]
        decoder_output = pad(decoder_output, self.tokenizer, self._max_words)
        decoder_input = pad(decoder_input, self.tokenizer, self._max_words)

        assert len(encoder_input) == self._max_words
        assert len(decoder_input) == self._max_words
        assert len(decoder_output) == self._max_words
        
        return {
            'encoder_input': np.asarray(encoder_input),
            'decoder_input': np.asarray(decoder_input),
            'decoder_output': np.asarray(decoder_output),
        }
