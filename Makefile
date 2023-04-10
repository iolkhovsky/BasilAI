RAW_DATA := 'data/chat.json'
TARGET := 'target_id'
DATASET := 'data/dataset.csv'
TRAIN_CONFIG := 'config/train.yaml'

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
