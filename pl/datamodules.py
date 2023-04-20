from typing import Any, Dict, Optional

import lightning as pl
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from utils import instantiate


class BasilAIDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.dataset_config = config.get("dataset", None)
        if self.dataset_config is None:
            raise ValueError(f"There is no 'dataset' configuration:\n{config}")
        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        dataset = instantiate(
            self.dataset_config, tokenizer=self.trainer.lightning_module.tokenizer
        )
        valid_share = self.dataset_config.get("valid_share", 0.05)
        valid_size = int(valid_share * len(dataset))
        train_size = len(dataset) - valid_size
        self.train_dataset, self.valid_dataset = random_split(
            dataset, [train_size, valid_size]
        )

    def train_dataloader(self) -> DataLoader:
        batch_size = self.dataset_config.get("train_batch", 64)
        num_workers = self.dataset_config.get("num_workers", 0)
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        batch_size = self.dataset_config.get("valid_batch", 16)
        num_workers = self.dataset_config.get("num_workers", 0)
        dataloader = DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return dataloader
