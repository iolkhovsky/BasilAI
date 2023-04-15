import os
import random
import telebot

from models import InferenceModel
from utils import read_yaml


GROUP_TYPE = 'group'
MY_NAME = '@NeuralGrayBot'
CONFIG_CMD = '/config'
GROUP_RESPONSE_PROB = 0.1


token = os.getenv('BOT_TOKEN', default = 'TOKEN')
bot = telebot.TeleBot(token)
config = read_yaml('config/eval.yaml')
model = InferenceModel(
	model_config=config['model'],
	tokenizer_config=config['tokenizer'],
)


def parse(msg_text):
	try:
		pars = {}
		msg_text = msg_text.replace(MY_NAME, '').replace(CONFIG_CMD, '')
		for item in msg_text.split(' '):
			if len(item) >= 3:
				key, value = item.split('=')
				pars[key] = value
		return pars
	except Exception as e:
		return {}


def config(pars):
	if 'p' in pars:
		GROUP_RESPONSE_PROB = float(pars['p'])


@bot.message_handler(commands=['start'])
def send_welcome(message):
	bot.reply_to(message, "Neural Gray Bot")


@bot.message_handler(func=lambda message: True)
def reply(message):
	msg_text = message.text
	if MY_NAME in msg_text and CONFIG_CMD in msg_text:
		pars = parse(msg_text)
		config(pars)
		bot.reply_to(message, f'Got new configuration: {pars}')
	else:
		if message.chat.type == GROUP_TYPE:
			if MY_NAME in message.text:
				bot.reply_to(message, model(message.text))
			elif random.uniform(0, 1) <= GROUP_RESPONSE_PROB:
				bot.send_message(message.chat.id, model(message.text))
		else:
			bot.send_message(message.chat.id, model(message.text))


if __name__ == '__main__':
	bot.infinity_polling()
