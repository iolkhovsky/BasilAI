import os
import random

import telebot

from bot import ChatBotRegistry
from models import InferenceModel
from utils import read_yaml


def register_handlers(app, bot):

	@bot.message_handler(commands=['start'])
	def welcome(message):
		bot.reply_to(message, app.welcome())

	@bot.message_handler(func=lambda message: message.reply_to_message and message.reply_to_message.from_user.id == telegram_bot.get_me().id)
	def on_reply(message):
		bot.reply_to(message, app.answer(message.text))

	@bot.message_handler(func=lambda message: True)
	def on_message(message):
		msg_text = message.text
		if app.try_parse_command(msg_text):
			bot.reply_to(message, f'Got new configuration: {str(app.config)}')
			return

		reaction = app.react(msg_text)
		if reaction is not None:
			pass  # TODO Support emoji reactions on messages

		if message.chat.type == 'group':
			if app.name in msg_text:
				bot.reply_to(message, app.answer(msg_text))
			elif random.uniform(0, 1) <= app.config.response_prob_in_group:
				bot.reply_to(message, app.answer(msg_text))
		else:
			bot.send_message(message.chat.id, app.answer(msg_text))


def run(app, bot):
	register_handlers(app, bot)
	telegram_bot.infinity_polling()


if __name__ == '__main__':
	config = read_yaml('config/eval.yaml')
	model = InferenceModel(
		model_config=config['model'],
		tokenizer_config=config['tokenizer'],
	)
	app = ChatBotRegistry.build('BasicChatBot', '@NeuralGrayBot', model)


	token = os.getenv('BOT_TOKEN', default = 'TOKEN')
	telegram_bot = telebot.TeleBot(token)

	run(app, telegram_bot)
