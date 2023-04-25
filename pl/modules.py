from typing import Any, Dict, Optional, Sequence, Union

import lightning as pl
import torch

from tokenizers import BaseTokenizer
from utils import instantiate, get_ram_consumption_mb


class BasilAIModule(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.model_config = config.get("model", None)
        self.tokenizer_config = config.get("tokenizer", None)
        self.optimizer_config = config.get("optimizer", None)
        self.scheduler_config = config.get("scheduler", None)
        self.dataset_config = config.get("dataset", None)
        self.criterion_config = config.get("criterion", None)
        if self.model_config is None:
            raise ValueError(f"There is no 'model' configuration:\n{config}")
        if self.tokenizer_config is None:
            raise ValueError(f"There is no 'tokenizer' configuration:\n{config}")
        if self.optimizer_config is None:
            raise ValueError(f"There is no 'optimizer' configuration:\n{config}")
        if self.dataset_config is None:
            raise ValueError(f"There is no 'dataset' configuration:\n{config}")
        if self.criterion_config is None:
            raise ValueError(f"There is no 'criterion' configuration:\n{config}")
        self.tokenizer: BaseTokenizer = instantiate(self.tokenizer_config)
        self.model = instantiate(
            self.model_config,
            start_token=self.tokenizer.start_token,
            stop_token=self.tokenizer.stop_token,
        )
        self.train_batch_size: Optional[int] = self.dataset_config.get(
            "train_batch", None
        )
        self.valid_batch_size: Optional[int] = self.dataset_config.get(
            "valid_batch", None
        )
        self.criterion = instantiate(
            self.criterion_config,
            masked_tokens=[self.tokenizer.pad_token, self.tokenizer.unk_token],
        )
        # self.save_hyperparameters(config)

    def forward(self, text_or_tokens):
        if isinstance(text_or_tokens, str):
            text_or_tokens = self.tokenizer.encode(text_or_tokens)
        input_tokens = torch.Tensor(text_or_tokens).long().unsqueeze(0).to(self.device)
        predicted_tokens = self.model(input_tokens)
        output_text = self.tokenizer.decode(predicted_tokens).replace("PAD", " ")
        return output_text

    def configure_optimizers(self) -> Any:
        optimizer = instantiate(self.optimizer_config, params=self.model.parameters())
        if self.scheduler_config is None:
            return optimizer
        scheduler = instantiate(self.scheduler_config, optimizer=optimizer)
        return [[optimizer], [scheduler]]

    def configure_callbacks(self) -> Union[Sequence[pl.Callback], pl.Callback]:
        pass

    def __step(self, batch: Dict[str, Any], batch_idx: int, stage: str) -> torch.Tensor:
        train_stage = stage == "train"

        in_tokens = batch["encoder_input"]
        dec_inputs = batch["decoder_input"]
        dec_targets = batch["decoder_output"]

        with torch.set_grad_enabled(train_stage):
            logits = self.model(tokens=in_tokens, dec_input=dec_inputs)

            b, n, c = logits.shape
            logits_reshaped = torch.reshape(logits, [b * n, -1])
            targets_reshaped = torch.reshape(dec_targets, [b * n]).long()

            loss = self.criterion(logits=logits_reshaped, targets=targets_reshaped)

        with torch.no_grad():
            logits_reshaped = logits_reshaped.clone().detach()
            predictions = torch.argmax(torch.softmax(logits_reshaped, dim=-1), dim=-1)
            mask = (targets_reshaped == self.tokenizer.unk_token) | (
                targets_reshaped == self.tokenizer.pad_token
            )
            accuracy = torch.mean(
                torch.eq(predictions[mask], targets_reshaped[mask]).float()
            )

        self.log_dict(
            {
                f"Loss/{stage}": loss,
                f"Accuracy/{stage}": accuracy,
                f"Resources/RAM": get_ram_consumption_mb(),
            },
            prog_bar=True,
            logger=True,
            on_step=train_stage,
            on_epoch=True,
            batch_size=self.train_batch_size if train_stage else self.valid_batch_size,
        )

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self.__step(batch, batch_idx, stage="train")
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        _ = self.__step(batch, batch_idx, stage="valid")

        in_tokens = batch["encoder_input"]
        text_input = batch["in_sentence"]
        target_output = batch["target_sentence"]

        with torch.no_grad():
            tokens = self.model(tokens=in_tokens).clone().detach().cpu()
            val_result = ""
            for sample_idx in range(len(in_tokens)):
                prediction = self.tokenizer.decode(tokens[sample_idx].tolist()).replace("PAD", " ")
                val_result += (
                    f"Sample #{sample_idx}:\n\ninput: {text_input[sample_idx]}\n\npredicted: "
                    f"{prediction}\n\ntarget: {target_output[sample_idx]}\n\n"
                )
            if len(val_result):
                self.logger.experiment.add_text(
                    f"Samples", val_result, global_step=self.global_step
                )
