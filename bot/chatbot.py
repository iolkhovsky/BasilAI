import random

from bot.registry import AbstractChatBot, ChatBotRegistry
from bot.config import BotConfig
from bot.queue import RandomAccessQueue


CONFIG_CMD = '/config'


@ChatBotRegistry.register('BasicChatBot')
class BasicChatBot(AbstractChatBot):
    def __init__(self, name, model, config=None):
        super().__init__()
        assert isinstance(name, str) and len(name) and '@' in name
        self._name = name
        assert callable(model)
        self._model = model
        self._config = BotConfig()
        if config:
            assert isinstance(config, BotConfig)
            self._config = config
        self._cache = RandomAccessQueue(
            max_size=self.config.message_cache_size
        )

    def answer(self, input_text) -> str:
        return self._model(input_text)

    def react(self, input_text) -> str:
        return None

    def remember(self, input_text):
        self._cache.push(input_text)

    def generate(self) -> str:
        if len(self._cache) < 1:
            return None
        context_text_id = random.randint(0, len(self._cache))
        return self._model(self._cache[context_text_id])

    def try_parse_command(self, input_text) -> bool:
        if CONFIG_CMD not in input_text:
            return False
        try:
            for s in [self._name, CONFIG_CMD]:
                input_text = input_text.replace(s, '')
            
            pars = {}
            for item in input_text.split(' '):
                if len(item) >= 3:
                    key, value = item.split('=')
                    pars[key] = value

            self._config.update(pars)
            return True
        except Exception as e:
            return False
    
    def welcome(self):
        return f'Hi! I am {self._name}!'

    @property
    def config(self) -> BotConfig:
        return self._config

    @property
    def name(self):
        return self._name
