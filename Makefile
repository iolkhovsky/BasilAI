RAW_DATA := 'data/chat.json'
TARGET := 'target_id'
DATASET := 'data/dataset.csv'
TRAIN_CONFIG := 'config/train.yaml'
TOKEN := 'chat_token'

install: FORCE
	python3 -m venv venv
	source venv/bin/activate
	pip3 install -r requirements.txt
FORCE:

dataset: FORCE
	python3 -m tools.generate_dataset \
		--path=${RAW_DATA} \
		--target-id=${TARGET} \
		--output=${DATASET}
FORCE:

train: FORCE
	python3 train.py \
		--config=${TRAIN_CONFIG}
FORCE:

bot: FORCE
	BOT_TOKEN=${TOKEN} python3 bot_server.py
FORCE:

run: FORCE
	python3 run_chat.py
FORCE:
