import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import read_json

from datetime import datetime

from itertools import chain, combinations


def parse_args():
    parser = argparse.ArgumentParser(prog="Extract dialog dataset from chat")
    parser.add_argument(
        "--path",
        default="data/datasets/raw/chat.json",
        type=str,
        help="Path to the raw chat json"
    )
    parser.add_argument("--user-id", type=str, help="Target user ID", default="user193042849")
    parser.add_argument(
        "--dialog-seconds-cooldown",
        type=int,
        help="Minimum time distance in seconds between different dialogs",
        default=300
    )
    parser.add_argument(
        "--dialog-memory",
        type=int,
        help="Message number in memory to answer",
        default=10
    )
    parser.add_argument(
        "--output",
        default="data/datasets/dataset_2",
        type=str,
        help="Path to the output folder",
    )
    args = parser.parse_args()
    return args


def load_data_frame(path: str):
    data = read_json(path)
    messages = data["messages"]
    df = pd.DataFrame(messages)
    df = df[df["type"] == "message"].reset_index(drop=True)
    df = df[~df["from"].isin(['EventPlannerChecker', 'PollBot'])].reset_index(drop=True)
    df = df[df["media_type"].astype(str).isin(['sticker', 'nan'])].reset_index(drop=True)
    df = df[df["file"].astype(str) == "nan"].reset_index(drop=True)
    df = df[df["thumbnail"].astype(str) == "nan"].reset_index(drop=True)
    df = df[df["photo"].astype(str) == "nan"].reset_index(drop=True)
    df = df[df["forwarded_from"].astype(str) == "nan"].reset_index(drop=True)
    df = df[df["location_information"].astype(str) == "nan"].reset_index(drop=True)
    df = df[df["poll"].astype(str) == "nan"].reset_index(drop=True)
    df = df[df["mime_type"].astype(str) == "nan"].reset_index(drop=True)
    df = df[df["via_bot"].astype(str) == "nan"].reset_index(drop=True)
    df["text_entities_n"] = df["text_entities"].apply(lambda x: len(x))
    df = df[df["text_entities_n"] == 1].reset_index(drop=True)
    df["text_type"] = df["text"].apply(lambda x: str(type(x)))
    df = df[df["text_type"] == "<class 'str'>"].reset_index(drop=True)
    df["media_type"] = df["media_type"].astype(str)
    df["sticker_emoji"] = df["sticker_emoji"].astype(str)
    df["reply_to_message_id"] = df["reply_to_message_id"].astype(str)
    df = df.drop(columns=["type", "actor", "actor_id", "action", "title", "members", "edited", "edited_unixtime",
                          "contact_information",
                          "file", "thumbnail", "photo", "forwarded_from", "location_information", "poll",
                          "mime_type", "via_bot",
                          "width", "height", "live_location_period_seconds", "duration_seconds", "message_id",
                          "text_entities_n", "text_entities", "text_type"])
    df = df.sort_values(by="date_unixtime")
    return df


def split_by_dialogs(df: pd.DataFrame, dialog_seconds_cooldown: int = -1):
    if dialog_seconds_cooldown < 0:
        dialogs = [0]*len(df)
    else:
        dialogs = []
        dialog = 0
        dt_prev = None
        for i, row in df.iterrows():
            dt_cur = datetime.fromisoformat(row["date"])
            if dt_prev is not None and (dt_cur - dt_prev).seconds > dialog_seconds_cooldown:
                dialog += 1
            dt_prev = dt_cur
            dialogs.append(dialog)
    df["dialog_id"] = dialogs
    return df


def get_dialog_df(df: pd.DataFrame, dialog_id: int) -> pd.DataFrame:
    dialog_df = df[df["dialog_id"] == dialog_id].reset_index(drop=True)
    dialog_df = dialog_df[["text", "from_id"]].rename(columns={"from_id": "user_id"})
    return dialog_df


def concat_user_messages(dialog_df: pd.DataFrame) -> pd.DataFrame:
    texts = []
    user_ids = []
    for i, row in dialog_df.iterrows():
        if len(texts) == 0 or row["user_id"] != user_ids[-1]:
            texts.append(row["text"])
            user_ids.append(row["user_id"])
        else:
            texts[-1] += " " + row["text"]
    dialog_df = pd.DataFrame({"text": texts, "user_id": user_ids})
    return dialog_df


def ordered_combinations(texts):
    def powerset(s):
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    idxs = list(range(len(texts)))
    combs = set([tuple(sorted(comb)) for comb in powerset(idxs) if len(comb)])
    combs = [". ".join([texts[i] for i in comb]) for comb in combs]
    return combs


def sample_dialog_questions_answers(
        dialog_df: pd.DataFrame,
        user_id: str = "",
        dialog_memory: int = -1
) -> pd.DataFrame:
    texts = dialog_df["text"].values
    user_ids = dialog_df["user_id"].values
    questions = [[]]
    answers = []
    for text, _user_id in zip(texts, user_ids):
        if (user_id == '' or _user_id == user_id) and len(questions[-1]):
            answers.append(text)
            questions.append(questions[-1].copy())
        questions[-1].append(text)
        if len(questions[-1]) > dialog_memory > 0:
            questions[-1] = questions[-1][1:]
    questions = questions[:len(answers)]

    res_questions = []
    res_answers = []
    for question, answer in zip(questions, answers):
        questions = ordered_combinations(
            question
        )
        res_questions.extend(questions)
        res_answers.extend([answer]*len(questions))

    res_df = pd.DataFrame({"question": res_questions, "answer": res_answers})
    return res_df


def sample_questions_answers(
        df: pd.DataFrame,
        user_id: str = "",
        dialog_memory: int = -1
) -> pd.DataFrame:
    sample_dfs = []
    for dialog_id in tqdm(np.unique(df["dialog_id"].values)):
        dialog_df = get_dialog_df(df, dialog_id=dialog_id)
        dialog_df = concat_user_messages(dialog_df)
        sample_df = sample_dialog_questions_answers(
            dialog_df,
            user_id=user_id,
            dialog_memory=dialog_memory
        )
        sample_dfs.append(sample_df)
    res_df = pd.concat(sample_dfs)
    return res_df


def get_dataset(
        path: str,
        user_id: str = "",
        dialog_seconds_cooldown: int = -1,
        dialog_memory: int = -1
) -> pd.DataFrame:
    df = load_data_frame(path)
    dialog_df = split_by_dialogs(df, dialog_seconds_cooldown=dialog_seconds_cooldown)
    res_df = sample_questions_answers(
        dialog_df,
        user_id=user_id,
        dialog_memory=dialog_memory
    )
    return res_df


def run(args):
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    df = load_data_frame(args.path)
    df.to_csv(os.path.join(output_dir, "dataset_original.csv"), index=False)
    dialog_df = split_by_dialogs(df, dialog_seconds_cooldown=args.dialog_seconds_cooldown)
    df.to_csv(os.path.join(output_dir, "dataset_dialogs.csv"), index=False)
    res_df = sample_questions_answers(
        dialog_df,
        user_id=args.user_id,
        dialog_memory=args.dialog_memory
    )
    print("Dataset size:", res_df.shape[0])
    res_df.to_csv(os.path.join(output_dir, "dataset.csv"), index=False)


if __name__ == "__main__":
    run(parse_args())
