import torch
import torchtext


def compute_accuracy(target_tokens, tokenizer, logits_or_scores=None, tokens=None):
    if tokens is None:
        assert logits_or_scores is not None
        assert len(logits_or_scores.shape) == 3, f'logits_or_scores should have shape (b, n c)'
        tokens = torch.argmax(logits_or_scores, dim=-1)
    assert len(target_tokens.shape) == 2, f'target_tokens should have shape (b, n)'
    assert len(tokens.shape) == 2, f'tokens should have shape (b, n)'

    masked_tokens = [tokenizer.unk_token, tokenizer.pad_token]
    mask = [x not in masked_tokens for x in target_tokens.flatten()]

    target_tokens_masked = target_tokens.flatten()[mask]
    predicted_tokens_nasked = tokens.flatten()[mask]

    if len(target_tokens_masked) < 1:
        return 0.

    return torch.mean(
        torch.eq(
            target_tokens_masked,
            predicted_tokens_nasked,
        ).float()
    )


def compute_bleu_score(target_tokens, tokenizer, logits_or_scores=None, tokens=None, max_n=4):
    if tokens is None:
        assert logits_or_scores is not None
        assert len(logits_or_scores.shape) == 3, f'logits_or_scores should have shape (b, n c)'
        tokens = torch.argmax(logits_or_scores, dim=-1)
    assert len(target_tokens.shape) == 2, f'target_tokens should have shape (b, n)'
    assert len(tokens.shape) == 2, f'tokens should have shape (b, n)'

    masked_tokens = [tokenizer.unk_token, tokenizer.pad_token]

    candidates_corpus, references_corpus = [], []
    for candidate, reference in zip(tokens, target_tokens):
        candidate = [tokenizer.decode_token(x) for x in candidate if x not in masked_tokens]
        reference = [tokenizer.decode_token(x) for x in reference if x not in masked_tokens]

        if tokenizer.stop_token_name in candidate:
            candidate = candidate[:candidate.index(tokenizer.stop_token_name)]
        if tokenizer.stop_token_name in reference:
            reference = reference[:reference.index(tokenizer.stop_token_name)]

        # there should be at least one token in each sample
        if len(candidate) == 0:
            candidate = ['candidate']
        if len(reference) == 0:
            reference = ['reference']

        candidates_corpus.append(candidate)
        references_corpus.append(reference)

    return torchtext.data.metrics.bleu_score(
        candidate_corpus=candidates_corpus,
        references_corpus=references_corpus,
        max_n=max_n,
        weights=[1./max_n] * max_n,
    )
