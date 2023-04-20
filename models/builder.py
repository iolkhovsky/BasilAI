import os

from torch import optim

import models


def build_model(model_config, optim_config):
    model_class = getattr(models, model_config["class"])
    checkpoint = model_config["checkpoint"]
    pars = model_config["parameters"]

    model = None
    if os.path.exists(checkpoint):
        pars["path"] = checkpoint
        model = model_class.load(**pars)
    else:
        model = model_class(**model_config["parameters"])

    optimizer = getattr(optim, optim_config["type"])(
        params=model.parameters(), **optim_config["parameters"]
    )
    return model, optimizer
