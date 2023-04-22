from typing import Any, Dict

import torch

from tokenizers import BaseTokenizer
from utils import instantiate


class InferenceModel:
    def __init__(self, config: Dict[str, Any]):
        self.device = torch.device(config.get("device", "cpu"))
        self.model_config = config.get("model", None)
        self.tokenizer_config = config.get("tokenizer", None)
        checkpoint_path = config.get("checkpoint", None)
        if self.model_config is None:
            raise ValueError(f"There is no 'model' configuration:\n{config}")
        if self.tokenizer_config is None:
            raise ValueError(f"There is no 'tokenizer' configuration:\n{config}")
        if checkpoint_path is None:
            raise ValueError(f"There is no 'checkpoint' configuration:\n{config}")
        self.tokenizer: BaseTokenizer = instantiate(self.tokenizer_config)
        self.model: torch.nn.Module = instantiate(
            self.model_config,
            start_token=self.tokenizer.start_token,
            stop_token=self.tokenizer.stop_token,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state_dict = {
            key.replace("model.", ""): val
            for key, val in checkpoint["state_dict"].items()
        }
        self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _postprocessor(text):
        return text.split("STOP", 1)[0].replace("PAD", "").replace("STOP", "")

    def predict(self, input_text: str):
        input_tokens = self.tokenizer.encode(input_text)
        input_tokens = (
            torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        )
        tokens = self.model(input_tokens)[0].cpu()
        text = self.tokenizer.decode(tokens.tolist())
        return InferenceModel._postprocessor(text)

    def __call__(self, *args, **kwds):
        return self.predict(*args, **kwds)
