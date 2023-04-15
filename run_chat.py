import os

from models import InferenceModel
from utils import read_yaml


config = read_yaml('config/eval.yaml')
model = InferenceModel(
	model_config=config['model'],
	tokenizer_config=config['tokenizer'],
)

while True:
    input_text = input('>\t: ')
    if input_text == 'exit':
        break
    response = model(input_text)
    print(f'<\t: {response}')
