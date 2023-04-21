import importlib
from typing import Any, Dict

__all_ = ["instantiate"]


def instantiate(config: Dict[str, Any], **additional_parameters: Any) -> Any:
    """
    Instantiate class by its config
    Args:
        config (Dict[str, Any]): configuration ('class' attribute is required)
        **additional_parameters: additional parameters to instantiate class

    Returns:
        class object
    """
    class_path = config.get("class", None)
    if class_path is None:
        raise ValueError(f"There is no 'class' attribute in instance config:\n{config}")
    module_name, class_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    parameters = config.get("parameters", {})
    parameters.update(additional_parameters)
    instance = cls(**parameters)
    return instance
