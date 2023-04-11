from collections import defaultdict
from enum import IntEnum
import pickle
import re
from tqdm import tqdm


class SpecToken(IntEnum):
    PAD = 0
    UNK = 1
    START = 2
    STOP = 3


class Tokenizer:
    def __init__(self, preprocessor=None, special_tokens=SpecToken, max_words=10000):
        self.preprocessor = preprocessor if preprocessor else self._default_preprocessor
        self.tokens = special_tokens
        self._word2id = None
        self._id2work = None
        self._max_words = max_words

    def encode(self, word):
        word = self.preprocessor(word)
        if word in self._word2id:
            return self._word2id[word]
        else:
            return int(self.tokens.UNK)

    def decode(self, token):
        token = int(token)
        if token in self._id2word:
            return self._id2word[token]
        else:
            return str(self.tokens.UNK)

    def encode_line(self, line):
        return [self.encode(x) for x in self.preprocessor(line).split(" ")]

    def decode_line(self, tokens):
        return " ".join([self.decode(x) for x in tokens])

    def fit(self, data, verbose=True):
        word2cnt = defaultdict(lambda: 0)
        word_list = self._extract_words(data)
        with tqdm(total=len(word_list)) as pbar:
            for idx, word in enumerate(word_list):
                if len(word) == 0:
                    continue
                word2cnt[word] += 1
                if idx % 10000 == 0:
                    pbar.set_description(f'Processed {idx + 1}/{len(word_list)}')
                pbar.update(1)
        loaded_cnt = len(word2cnt)
        if verbose:
            print(f'Loaded {loaded_cnt} unique words from {len(word_list)} corpus'
                  f' ({100. * loaded_cnt / len(word_list)} %)')
        words_limit = self._max_words - len(self.tokens)
        popular_words = sorted(
            word2cnt.items(), reverse=True, key=lambda x: x[1])[:words_limit]
        if verbose:
            print(f'20 most popular words:\n{popular_words[:20]}')

        self._word2id = {item.name: item.value for item in self.tokens}
        for word, _ in popular_words:
            self._word2id[word] = len(self._word2id)
        assert len(self._word2id) <= self._max_words
        self._id2word = {v: k for k, v in self._word2id.items()}

    def _extract_words(self, data):
        words = []
        if isinstance(data, list):
            for line in data:
                assert isinstance(line, str)
                words.extend([self.preprocessor(x) for x in line.split(' ')])
        else:
            assert isinstance(data, str)
            words.extend([self.preprocessor(x) for x in data.split(' ')])
        return words

    def __len__(self):
        return len(self._id2word)

    @staticmethod
    def _default_preprocessor(word):
        return re.sub('[^A-Za-zА-Яа-я\s]+', '', word).lower().strip()

    @property
    def start_token(self):
        return int(self.tokens.START)

    @property
    def stop_token(self):
        return int(self.tokens.STOP)

    @property
    def pad_token(self):
        return int(self.tokens.PAD)

    @property
    def unk_token(self):
        return int(self.tokens.UNK)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            assert isinstance(obj, Tokenizer)
            return obj
