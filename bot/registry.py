from abc import ABC, abstractmethod

from bot.config import BotConfig


class AbstractChatBot(ABC):
    @abstractmethod
    def answer(self, input_text) -> str:
        """
        Make direct response to the input text
        """
        pass

    @abstractmethod
    def react(self, input_text) -> str:
        """
        Make reaction to the input text
        """
        pass

    @abstractmethod
    def remember(self, input_text):
        """
        Put input text to the internal cache
        """
        pass

    @abstractmethod
    def generate(self) -> str:
        """
        Make random text without direct context (initiate a discussion)
        """
        pass

    @abstractmethod
    def try_parse_command(self, input_text) -> bool:
        """
        Try to parse input text to handle commands
        """
        pass

    @abstractmethod
    def welcome(self) -> str:
        """
        Introduce yourself
        """
        pass

    @property
    @abstractmethod
    def config(self) -> BotConfig:
        """
        Get actual BotConfig
        """
        pass


class ChatBotRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(reg_cls):
            assert issubclass(reg_cls, AbstractChatBot)
            ChatBotRegistry._registry[name] = reg_cls
            return reg_cls
        return decorator

    @classmethod
    def build(cls, name, *args, **kwargs):
        cls = ChatBotRegistry._registry.get(name)
        if cls is not None:
            return cls(*args, **kwargs)
        raise ValueError(f'No registered implementation of "{name}"')
