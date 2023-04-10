import json


def read_json(path):
    with open(path, 'rt') as f:
        return json.load(f)
