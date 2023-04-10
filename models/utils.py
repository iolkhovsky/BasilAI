import torch


def infer(model, input_tokens_batch, tokenizer):
    model.eval()
    inputs, outputs = [], []
    for input_tokens in input_tokens_batch:
        inputs.append(tokenizer.decode_line(input_tokens))
        input_tokens = input_tokens.unsqueeze(0)
        prediction = model(input_tokens)
        if input_tokens.device != torch.device('cpu'):
            prediction = [x.cpu() for x in prediction]
        outputs.append(tokenizer.decode_line(prediction))
    return inputs, outputs
