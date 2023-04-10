import re


def preprocess_word(word):
    return re.sub('[^A-Za-zА-Яа-я\s]+', '', word).lower().strip()
