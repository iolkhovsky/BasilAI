import torch

import models
from datasets import Tokenizer


class InferenceModel:
    def __init__(self, model_config, tokenizer_config, device='cpu'):
        model_class = getattr(models, model_config['class'])
        checkpoint = model_config['checkpoint']
        pars = model_config['parameters']
        pars['path'] = checkpoint
        self._device = torch.device(device)
        self._core = model_class.load(**pars).to(self._device)
        self._core.eval()
        self._tokenizer = Tokenizer.load(tokenizer_config['path'])

    @staticmethod
    def _postprocessor(text):
        return text.replace('PAD', '').replace('STOP', '')

    def predict(self, input_text):
        prediction = self._core.infer(input_text, self._tokenizer)
        return InferenceModel._postprocessor(prediction)

    def __call__(self, *args, **kwds):
        return self.predict(*args, **kwds)
