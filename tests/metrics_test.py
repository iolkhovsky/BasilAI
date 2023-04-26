import pytest
import torch

from utils import compute_accuracy, compute_bleu_score, pad
from tokenizers import SimpleTokenizer


DICTIONARY = ['Computes the BLEU score between a candidate translation corpus and a references translation corpus']
TARGET_SENTENCES = (
    'Computes the BLEU score between',
    'a candidate translation corpus and a references'
)
PREDICTED_SENTENCES = (
    'Someone Computes the BLEU score between',
    'a candidate interpretation sequence or a references',
)


def test_accuracy():
    tokenizer = SimpleTokenizer(path='')
    tokenizer.fit(DICTIONARY)

    target_batch = [tokenizer.encode(x) for x in TARGET_SENTENCES]
    predicted_batch = [tokenizer.encode(x) for x in PREDICTED_SENTENCES]

    target_batch = [pad(x, tokenizer, 10) for x in target_batch]
    predicted_batch = [pad(x, tokenizer, 10) for x in predicted_batch]

    acc = compute_accuracy(
        target_tokens=torch.tensor(target_batch, dtype=torch.int),
        tokenizer=tokenizer,
        logits_or_scores=None,
        tokens=torch.tensor(predicted_batch, dtype=torch.int),
    )

    target_acc = 0.
    total_tokens = 0
    masked_tokens = [tokenizer.pad_token, tokenizer.stop_token]
    for target_phrase, predicted_phrase in zip(target_batch, predicted_batch):
        for target_token, predicted_token in zip(target_phrase, predicted_phrase):
            if target_token not in masked_tokens:
                total_tokens += 1
                target_acc += target_token == predicted_token
    target_acc /= total_tokens

    assert acc == pytest.approx(target_acc)


def test_bleu():
    # TODO
    pass