import os
import random

import telebot

from models import InferenceModel
from utils import read_yaml

GROUP_TYPE = "group"
MY_NAME = "@NeuralGrayBot"
CONFIG_CMD = "/config"


class BotConfig:
    group_response_prob = 0.1

    @staticmethod
    def parse(text):
        pars = {}
        msg_text = text.replace(MY_NAME, "").replace(CONFIG_CMD, "")
        for item in msg_text.split(" "):
            if len(item) >= 3:
                key, value = item.split("=")
                pars[key] = value
        return pars

    @staticmethod
    def try_parse(text):
        try:
            pars = BotConfig.parse(text)
            if "p" in pars:
                BotConfig.group_response_prob = float(pars["p"])
            return f"New configuration: {BotConfig.group_response_prob}"
        except Exception as e:
            return f"Couldnt update configuration"


token = os.getenv("BOT_TOKEN", default="TOKEN")
bot = telebot.TeleBot(token)
config = read_yaml("config/eval.yaml")
model = InferenceModel(config)
config = BotConfig()


@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(message, "Neural Gray Bot")


@bot.message_handler(func=lambda message: True)
def reply(message):
    msg_text = message.text
    if MY_NAME in msg_text and CONFIG_CMD in msg_text:
        bot.reply_to(message, config.try_parse(msg_text))
    else:
        if message.chat.type == GROUP_TYPE:
            if MY_NAME in message.text:
                bot.reply_to(message, model(message.text))
            elif random.uniform(0, 1) <= config.group_response_prob:
                bot.send_message(message.chat.id, model(message.text))
        else:
            bot.send_message(message.chat.id, model(message.text))


if __name__ == "__main__":
    bot.infinity_polling()
