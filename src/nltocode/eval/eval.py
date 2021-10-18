import argparse
import json
import re

import pandas as pd

import bleu_score as bls


def compute_best_bleu_score(expected, predicted):
    if not predicted:
        return ([''], 0.0)

    prediction_bls_list = []

    for prediction in predicted:
        bleu, precisions, bp, ratio, translation_length, reference_length = bls.compute_bleu([[expected]],
                                                                                             [prediction],
                                                                                             max_order=4,
                                                                                             smooth=False)
        prediction_bls_list.append((prediction, bleu))

    best_predicted = max(prediction_bls_list, key=lambda tup: tup[1])

    return best_predicted


def compute_best_predicted_bls(data, expected_column, predictions_column, k=10):
    data['best_predicted_code_tokenized_bls'] = data.apply(
        lambda row: compute_best_bleu_score(row[expected_column], row[predictions_column][:k]), axis=1)

    bleu, precisions, bp, ratio, translation_length, reference_length = bls.compute_bleu(
        [[exp] for exp in data[expected_column]], [pred[0] for pred in data['best_predicted_code_tokenized_bls']],
        max_order=4, smooth=False)

    return bleu


def tokenize_for_bleu_eval(code):
    if code is None:
        return None
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    return tokens


def compute_num_invalid_predictions(data):
    num_null_predictions = data['predicted_codesnippets'].apply(
        lambda predictions: len([p for p in predictions if p is None]))
    return int(num_null_predictions.sum())


def compute_num_valid_predictions(data):
    num_non_null_predictions = data['predicted_codesnippets'].apply(
        lambda predictions: len([p for p in predictions if p is not None]))
    return int(num_non_null_predictions.sum())


def tokenize_data(data):
    data['expected_codesnippet_tokenized'] = data['expected_codesnippet'].apply(
        lambda expected_codesnippet: tokenize_for_bleu_eval(expected_codesnippet))

    data['predicted_codesnippets_tokenized'] = data['predicted_codesnippets'].apply(
        lambda predicted_codesnippets: [tokenize_for_bleu_eval(predicted_codesnippet) for predicted_codesnippet in
                                        predicted_codesnippets if predicted_codesnippet])


def load_data(path):
    is_jsonl = path.lower().endswith('.jsonl')
    return pd.read_json(path, lines=is_jsonl, dtype=False)


def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def get_args():
    parser = argparse.ArgumentParser("Evaluate predicted target file", fromfile_prefix_chars='@')

    parser.add_argument("--target-input-file", type=str, default="target.jsonl")
    parser.add_argument("--eval-output-file", type=str, default=None)

    return parser.parse_args()


def main():
    args = get_args()
    print("Eval args:", vars(args))

    data = load_data(args.target_input_file)

    tokenize_data(data)

    best_20_predicted_code_bls = compute_best_predicted_bls(data, 'expected_codesnippet_tokenized',
                                                            'predicted_codesnippets_tokenized', k=20)
    best_10_predicted_code_bls = compute_best_predicted_bls(data, 'expected_codesnippet_tokenized',
                                                            'predicted_codesnippets_tokenized', k=10)
    best_3_predicted_code_bls = compute_best_predicted_bls(data, 'expected_codesnippet_tokenized',
                                                           'predicted_codesnippets_tokenized', k=3)
    top_predicted_code_bls = compute_best_predicted_bls(data, 'expected_codesnippet_tokenized',
                                                        'predicted_codesnippets_tokenized', k=1)
    num_valid_predictions = compute_num_valid_predictions(data)
    num_invalid_predictions = compute_num_invalid_predictions(data)

    if args.eval_output_file is not None:
        eval_results = {
            'top_predicted_code_bls': top_predicted_code_bls,
            # 'top_exact_match_accuracy': top_exact_match_accuracy,
            # 'exact_matches': exact_matches,
            # 'top_token_match_accuracy': top_token_match_accuracy,
            'best_3_predicted_code_bls': best_3_predicted_code_bls,
            'best_10_predicted_code_bls': best_10_predicted_code_bls,
            'best_20_predicted_code_bls': best_20_predicted_code_bls,
            'num_valid_predictions': num_valid_predictions,
            'num_invalid_predictions': num_invalid_predictions
        }
        save_data(eval_results, args.eval_output_file)
    else:
        print("data = load_data('" + args.target_input_file + "')")
        print('GLOBAL TOP 20 PREDICTED CODE BLEU SCORE', best_20_predicted_code_bls)
        print('GLOBAL TOP 10 PREDICTED CODE BLEU SCORE', best_10_predicted_code_bls)
        print('GLOBAL TOP 3 PREDICTED CODE BLEU SCORE', best_3_predicted_code_bls)
        print('GLOBAL TOP PREDICTED CODE BLEU SCORE', top_predicted_code_bls)
        # print('TOP EXACT MATCH ACCURACY', top_exact_match_accuracy)
        # print('TOP TOKEN MATCH ACCURACY', top_token_match_accuracy)
        # print('Examples with exact match:', exact_matches)


if __name__ == '__main__':
    main()
