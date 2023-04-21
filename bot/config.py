import yaml
from dataclasses import dataclass
from typing import Type, TypeVar, Dict


T = TypeVar('T')


@dataclass
class BotConfig:
    response_prob_in_group: float = 0.1
    message_cache_size: int = 20

    @staticmethod
    def from_yaml(file_path: str) -> 'BotConfig':
        with open(file_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        return BotConfig(**yaml_data)

    def update(self, data: Dict[str, str]):
        for key, value in data.items():
            if hasattr(self, key):
                attr_type = type(getattr(self, key))
                setattr(self, key, attr_type(value))
