import json
import pickle


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)
