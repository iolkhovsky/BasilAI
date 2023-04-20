import argparse
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import preprocess_word, read_json


def parse_args():
    parser = argparse.ArgumentParser(prog="Extract dialog dataset from chat")
    parser.add_argument(
        "--path", default="data/chat.json", type=str, help="Path to the raw chat json"
    )
    parser.add_argument("--target-id", type=str, help="Target person ID")
    parser.add_argument(
        "--output",
        default="data/dataset.csv",
        type=str,
        help="Path to the output dataset",
    )
    args = parser.parse_args()
    return args


def run(args):
    raw_data = read_json(args.path)
    assert "messages" in raw_data
    messages = raw_data["messages"]
    print(f"Loaded {len(messages)} messages from {args.path}")

    id2msg = {msg["id"]: msg for msg in messages if "id" in msg}

    def message_filter(msg):
        if "from_id" not in msg:
            return False
        if msg["from_id"] != args.target_id:
            return False
        if "text" not in msg:
            return False
        if not isinstance(msg["text"], str):
            return False
        if "reply_to_message_id" not in msg:
            return False
        question_msg = id2msg[msg["reply_to_message_id"]]
        if "text" not in question_msg:
            return False
        if not isinstance(question_msg["text"], str):
            return False
        if question_msg["from_id"] == args.target_id:
            return False
        if len(question_msg["text"]) == 0:
            return False
        if len(msg["text"]) == 0:
            return False
        return True

    filtered_messages = list(filter(message_filter, messages))
    df_data = defaultdict(list)
    with tqdm(total=len(filtered_messages)) as pbar:
        for msg in filtered_messages:
            df_data["answer"].append(msg["text"].replace(",", ""))
            df_data["question"].append(
                id2msg[msg["reply_to_message_id"]]["text"].replace(",", "")
            )
            pbar.update(1)

    df = pd.DataFrame.from_dict(df_data)
    df.to_csv(args.output, index=False)
    print(f"{len(df)} samples saved to {args.output}")

    questions = df_data["question"]
    answers = df_data["answer"]

    print(f"Data stats:")
    print(f"Question length q90: {np.quantile([len(x) for x in questions], 0.9)}")
    print(f"Answer length q90: {np.quantile([len(x) for x in answers], 0.9)}")

    vocabulary = defaultdict(int)
    for sentence in questions + answers:
        for word in sentence.strip().split(" "):
            vocabulary[preprocess_word(word)] += 1

    print(f"Unique words: {len(vocabulary)}")
    vocabulary = {k: v for k, v in vocabulary.items() if len(k) > 3}
    frequent_words = sorted(vocabulary.items(), reverse=True, key=lambda x: x[1])[:30]
    print(f"Most frequent long words:")
    for word, freq in frequent_words:
        print(f"\t<{word}> : {freq}")


if __name__ == "__main__":
    run(parse_args())
