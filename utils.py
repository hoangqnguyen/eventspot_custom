import time
import os
import re
import json
import pickle
import gzip


def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)


def load_gz_json(fpath):
    with gzip.open(fpath, "rt", encoding="ascii") as fp:
        return json.load(fp)


def store_json(fpath, obj, pretty=False):
    kwargs = {}
    if pretty:
        kwargs["indent"] = 2
        kwargs["sort_keys"] = True
    with open(fpath, "w") as fp:
        json.dump(obj, fp, **kwargs)


def store_gz_json(fpath, obj):
    with gzip.open(fpath, "wt", encoding="ascii") as fp:
        json.dump(obj, fp)


def load_pickle(fpath):
    with open(fpath, "rb") as fp:
        return pickle.load(fp)


def store_pickle(fpath, obj):
    with open(fpath, "wb") as fp:
        pickle.dump(obj, fp)


def load_text(fpath):
    lines = []
    with open(fpath, "r") as fp:
        for l in fp:
            l = l.strip()
            if l:
                lines.append(l)
    return lines


def store_text(fpath, s):
    with open(fpath, "w") as fp:
        fp.write(s)


def clear_files(dir_name, re_str, exclude=[]):
    for file_name in os.listdir(dir_name):
        if re.match(re_str, file_name):
            if file_name not in exclude:
                file_path = os.path.join(dir_name, file_name)
                os.remove(file_path)


class Timer:
    def __init__(self, message="Elapsed time"):
        self.message = message

    def __enter__(self):
        self.start_time = time.time()
        return self  # If you need to return any object, it would be here

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.time() - self.start_time
        print(f"{self.message}: {elapsed_time:.4f} seconds")
