import re


def preprocess_word(word):
    return re.sub('[^A-Za-zА-Яа-я\s]+', '', word).lower().strip()


def pad(seq, tokenizer, size, prepadding=False):
    pad_length = size - len(seq)
    if pad_length == 0:
        return seq
    if prepadding:
        return pad_length * [tokenizer.pad_token] + seq
    else:
        return seq + pad_length * [tokenizer.pad_token]
