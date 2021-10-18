import os

import pandas as pd


def load_json(path):
    if not os.path.isfile(path):
        raise ValueError("'%s' doesn't exist or is no file" % path)
    is_jsonl = path.lower().endswith('.jsonl')
    return pd.read_json(path, lines=is_jsonl, dtype=False)


def save_json(dataframe, filename):
    is_jsonl = filename.endswith('.jsonl')
    dataframe.to_json(filename, orient='records', lines=is_jsonl)
    append_nl(filename)


def append_nl(filename):
    with open(filename, 'rb+') as f:
        last_line = get_last_line(f)
        if not last_line.endswith('\n'):
            f.write(bytes('\n', 'UTF-8'))


def get_last_line(f):
    f.seek(0, os.SEEK_END)
    while f.read(1) != b'\n':
        f.seek(-2, os.SEEK_CUR)
    last_line = f.readline().decode()
    return last_line
