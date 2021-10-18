import argparse
from collections import Counter

from nltocode.preprocessing.filehandling import load_json


def get_args():
    parser = argparse.ArgumentParser("Preproc data")
    parser.add_argument("preproc_data_path", type=str)
    return parser.parse_args()


def main():
    args = get_args()
    preproc_df = load_json(args.preproc_data_path)

    nl_counter = Counter(preproc_df['nl_enc'].apply(lambda nl_enc: len(nl_enc)))
    nl_histogram = dict(sorted(nl_counter.items()))
    print("NL:", nl_histogram)
    print("Cumul. NL:", accumulate(nl_counter))

    ast_seq_counter = Counter(preproc_df['ast_seq_enc'].apply(lambda ast_seq_enc: len(ast_seq_enc)))
    ast_seq_histogram = dict(sorted(ast_seq_counter.items()))

    print("AST Seqs:", ast_seq_histogram)
    print("Cumul. AST Seqs:", accumulate(ast_seq_counter))


def accumulate(counter):
    out = {}
    sum = 0
    for i, k in sorted(counter.items()):
        sum += k
        out[i] = sum
    return out


if __name__ == '__main__':
    main()
