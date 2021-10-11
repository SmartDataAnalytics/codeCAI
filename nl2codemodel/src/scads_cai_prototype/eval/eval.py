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

# def compute_top_exact_match_accuracy(data, expected_column, predictions_column):
#     has_top_exact_match = data.apply(lambda row: row_has_top_exact_match(row[expected_column], row[predictions_column]),
#                                      axis=1)
#     num_top_exact_match = has_top_exact_match.sum()
#     num_samples = len(data)
#
#     top_exact_match_accuracy = float(num_top_exact_match) / num_samples
#
#     # print(data[has_top_exact_match].to_json(lines=True, orient='records'))
#     return top_exact_match_accuracy, data[has_top_exact_match]['expected_codesnippet'].to_list()
#
#
# def row_has_top_exact_match(expected, sorted_predictions):
#     return (len(sorted_predictions) > 0) and (expected == sorted_predictions[0])
#
#
# def compute_top_token_match_accuracy(data, expected_column, predictions_column):
#     df = pd.DataFrame()
#     df[['num_top_matching_tokens', 'top_matching_tokens_weight']] = data.apply(
#         lambda row: pd.Series(compute_top_matching_tokens(row[expected_column], row[predictions_column])), axis=1)
#     top_token_match_accuracy = float(df['num_top_matching_tokens'].sum()) / df['top_matching_tokens_weight'].sum()
#
#     return top_token_match_accuracy
#
#
# def compute_top_matching_tokens(expected, sorted_predictions):
#     top_predicted = sorted_predictions[0] if sorted_predictions else []
#     matches = [1 for expected_token, predicted_token in zip(expected, top_predicted) if
#                expected_token == predicted_token]
#     num_top_matching_tokens = sum(matches)
#     top_matching_tokens_weight = max(len(expected), len(top_predicted))
#     return num_top_matching_tokens, top_matching_tokens_weight


# def compute_top_predicted_bls(expectations, sorted_predictions, empty_default=''):
#     expected = [[exp] for exp in expectations]
#     top_predicted = [pred[0] if (len(pred) > 0) else [empty_default] for pred in sorted_predictions]
#     bleu, precisions, bp, ratio, translation_length, reference_length = bls.compute_bleu(
#         expected, top_predicted, max_order=4, smooth=False)
#     return bleu


# data = load_data('/home/ubuntu/Documents/scads_cai_data/target_y_curated_v0_149.jsonl')
# data = load_data('/home/ubuntu/Documents/scads_cai_data/target_curated_y_25_v349.jsonl')
# data = load_data('/home/ubuntu/Documents/scads_cai_data/target_y_curated_v0_149_sample.jsonl')
# data = load_data('/home/ubuntu/Documents/scads_cai_data/curated/target_y.jsonl')
# data = load_data('/home/ubuntu/Documents/scads_cai_data/curated/target_y_v0_c149_25r_200l_20b.jsonl')
# GLOBAL TOP 20 PREDICTED CODE BLEU SCORE 0.2416299017879911
# GLOBAL TOP 10 PREDICTED CODE BLEU SCORE 0.21361790116371543
# GLOBAL TOP 3 PREDICTED CODE BLEU SCORE 0.1639708771809752
# GLOBAL TOP PREDICTED CODE BLEU SCORE 0.12092255133034163
# data = load_data('/home/ubuntu/Documents/scads_cai_data/curated/target_y_v7_c349_25r_200l_20b.jsonl')
# GLOBAL TOP 20 PREDICTED CODE BLEU SCORE 0.26092932988958983
# GLOBAL TOP 10 PREDICTED CODE BLEU SCORE 0.235872280482935
# GLOBAL TOP 3 PREDICTED CODE BLEU SCORE 0.20181446927592836
# GLOBAL TOP PREDICTED CODE BLEU SCORE 0.17229562028973294
# data = load_data('/home/ubuntu/Documents/scads_cai_data/curated/target_y_curated_20200829_v0_c991_20r_200l_20b.jsonl')
# GLOBAL TOP 20 PREDICTED CODE BLEU SCORE 0.25922217440886974
# GLOBAL TOP 10 PREDICTED CODE BLEU SCORE 0.24073880314050278
# GLOBAL TOP 3 PREDICTED CODE BLEU SCORE 0.20549350126242122
# GLOBAL TOP PREDICTED CODE BLEU SCORE 0.17346888089808113


# Module(body=[Expr(value=Call(func=Attribute(value=Name(id='plt'),attr='savefig'),args=[Num(n=60)],keywords=[]))])
# Module(body=[Expr(value=Call(func=Attribute(value=Name(id='plt'),attr='savefig'),args=[Num(n=60)],keywords=[]))])
#########################################################################################


# compute_top_predicted_bls(data['expected_codesnippet_tokenized'], data['predicted_codesnippets_tokenized'])

# print()
# print(data.to_string())
# print()
# print(data.head())
# print()
# top_exact_match_accuracy, exact_matches = compute_top_exact_match_accuracy(data, 'expected_codesnippet_tokenized','predicted_codesnippets_tokenized')
# top_token_match_accuracy = compute_top_token_match_accuracy(data, 'expected_codesnippet_tokenized','predicted_codesnippets_tokenized')
