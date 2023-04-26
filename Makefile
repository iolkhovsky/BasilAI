RAW_DATA := 'data/datasets/raw/chat.json'
TARGET := 'target_id'
DATASET := 'data/datasets/dataset_1/dataset.csv'
TRAIN_CONFIG := 'config/train.yaml'
TOKEN := 'chat_token'

RAW_DATASET := 'data/datasets/raw/chat.json'
USER_ID := 'user193042849'
DIALOG_SECONDS_COOLDOWN := '300'
DIALOG_MEMORY := '10'
DATASET_OUTPUT := 'data/datasets/dataset_2'
TOKENIZER_DATASET := 'data/datasets/dataset_2/dataset_original.csv'
TOKENIZER := 'tokenizers.SimpleTokenizer'
TOKENIZER_SIZE := '10000'
TOKENIZER_OUTPUT := 'data/tokenizers/tokenizer_2/vocab.json'

install-dev: FORCE
	python3 -m venv venv
	source venv/bin/activate
	pip3 install -r dev_requirements.txt
FORCE:

install-prod: FORCE
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

dataset_advanced: FORCE
	python3 -m tools.generate_advanced_dataset \
		--user-id=${USER_ID} \
		--path=${RAW_DATASET} \
		--dialog-seconds-cooldown=${DIALOG_SECONDS_COOLDOWN} \
		--dialog-memory=${DIALOG_MEMORY} \
		--output=${DATASET_OUTPUT}
FORCE:

fit_tokenizer: FORCE
	python3 -m tools.fit_tokenizer \
		--dataset=${TOKENIZER_DATASET} \
		--tokenizer=${TOKENIZER} \
		--size=${TOKENIZER_SIZE} \
		--output=${TOKENIZER_OUTPUT}
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
