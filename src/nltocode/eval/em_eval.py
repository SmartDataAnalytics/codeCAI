import argparse
import ast
import json
from typing import List

import astor
import pandas as pd


def compute_best_exact_match_accuracy(expected, predicted: List):
    return any(prediction == expected for prediction in predicted)


def compute_top_k_exact_match_accuracy(data, expected_column, predictions_column, k=1):
    has_match: pd.Series = data.apply(
        lambda row: compute_best_exact_match_accuracy(row[expected_column], row[predictions_column][:k]),
        axis=1)
    return has_match.mean()


def load_data(path):
    is_jsonl = path.lower().endswith('.jsonl')
    return pd.read_json(path, lines=is_jsonl, dtype=False)


def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def get_args():
    parser = argparse.ArgumentParser("Evaluate predicted target file", fromfile_prefix_chars='@')

    parser.add_argument("--target-input-file", type=str, default="target.jsonl")
    parser.add_argument("--mode", choices=['snippet', 'ast_seq'], default="snippet")
    parser.add_argument("--em-eval-output-file", type=str, default=None)

    return parser.parse_args()


def reformat(snippet):
    try:
        return astor.to_source(ast.parse(snippet))
    except:
        print("Error in snippet, not reformatting:\n%s" % snippet)
        return snippet


def main():
    args = get_args()
    print("Eval args:", vars(args))

    data = load_data(args.target_input_file)

    if args.mode == 'snippet':
        data['expected_codesnippet_reformatted'] = data['expected_codesnippet'].apply(reformat)
        expected_column = 'expected_codesnippet_reformatted'
        predicted_column = 'predicted_codesnippets'
    elif args.mode == 'ast_seq':
        expected_column = 'expected_ast_seq'
        predicted_column = 'predicted_ast_seqs'
    else:
        raise ValueError('Unknown mode %s' % args.mode)

    exact_match_accuracy = compute_top_k_exact_match_accuracy(data, expected_column, predicted_column, k=1)
    top_3_match_accuracy = compute_top_k_exact_match_accuracy(data, expected_column, predicted_column, k=3)
    top_10_match_accuracy = compute_top_k_exact_match_accuracy(data, expected_column, predicted_column, k=10)

    eval_results = {
        'exact_match_accuracy': exact_match_accuracy,
        'top_3_match_accuracy': top_3_match_accuracy,
        'top_10_match_accuracy': top_10_match_accuracy
    }

    save_data(eval_results, args.em_eval_output_file)


if __name__ == '__main__':
    main()
