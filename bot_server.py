import os
import telebot

from models import InferenceModel
from utils import read_yaml


token = os.getenv('BOT_TOKEN', default = 'TOKEN')
bot = telebot.TeleBot(token)
config = read_yaml('config/eval.yaml')
model = InferenceModel(
	model_config=config['model'],
	tokenizer_config=config['tokenizer'],
)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
	bot.reply_to(message, "Neural Gray Bot")


@bot.message_handler(func=lambda message: True)
def reply(message):
	response = model(message.text)
	bot.reply_to(message, response)


if __name__ == '__main__':
	bot.infinity_polling()
