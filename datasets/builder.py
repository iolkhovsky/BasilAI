import torch
from torch.utils.data import DataLoader

import datasets
from datasets import Tokenizer


def build_dataloaders(config):
    tokenizer = Tokenizer()

    dataset_class = getattr(datasets, config['class'])
    pars = config['parameters']
    pars['tokenizer'] = tokenizer
    pars['fit_tokenizer'] = True

    dataset = dataset_class(**pars)

    val_size = int(config['val_share'] * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    return DataLoader(train_dataset, batch_size=config['train_batch'], shuffle=True), \
        DataLoader(val_dataset, batch_size=config['val_batch'], shuffle=True), \
        tokenizer
