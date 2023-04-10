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
    ):
        self.tokenizer = tokenizer
        self._max_words = max_words
        self._df = pd.read_csv(path)
       
        if fit_tokenizer:
            self.tokenizer.fit(self._df['answers'] + self._df['questions'])

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):       
        in_sentence = self._df['questions'][index]
        encoder_input = self.tokenizer.encode_line(in_sentence)[:self._max_words]
        encoder_input = pad(encoder_input, self.tokenizer, self._max_words, prepadding=True)

        target_sentence = self._df['answers'][index]
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
