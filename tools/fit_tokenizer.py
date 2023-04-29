import argparse
import os

import pandas as pd

from utils import instantiate


def parse_args():
    parser = argparse.ArgumentParser(prog="Fit tokenizer")
    parser.add_argument(
        "--dataset", type=str, default="data/datasets/dataset_2/dataset.csv", help="Path to the dataset"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="tokenizers.SimpleTokenizer", help="Tokenizer class"
    )
    parser.add_argument(
        "--size", type=int, default=10000, help="Tokenizer vocab size"
    )
    parser.add_argument(
        "--output", type=str, default="data/tokenizers/tokenizer_2/vocab.json", help="Path to save tokenizer",
    )
    args = parser.parse_args()
    return args


def run(args):
    output_dir = os.path.split(args.output)[0]
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = instantiate({"class": args.tokenizer}, max_words=args.size)
    df = pd.read_csv(args.dataset)
    if "text" in df:
        all_sentences = df["text"].values.tolist()
    elif "question" in df and "answer" in df:
        all_sentences = df["question"].values.tolist() + df["answer"].values.tolist()
    else:
        raise ValueError(f"Invalid dataset schema: '{df.columns}'")
    tokenizer.fit(all_sentences)
    tokenizer.save(args.output)


if __name__ == "__main__":
    run(parse_args())
