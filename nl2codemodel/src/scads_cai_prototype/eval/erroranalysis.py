import argparse
import math
from collections import Counter
from functools import reduce

from scads_cai_prototype.preprocessing.filehandling import load_json


def comp_recall_seq(data):
    data['replaced_percentage_pref_exp'] = data.apply(
        lambda row: comp_fraction_prefix_seq(row['replaced_common_prefix_len'],
                                             row['replaced_filtered_expected_ast_seq_len']),
        axis=1)

    data['percentage_pref_exp'] = data.apply(
        lambda row: comp_fraction_prefix_seq(row['common_prefix_len'], row['expected_ast_seq_len']),
        axis=1)

    return data['replaced_percentage_pref_exp'].mean(), data['percentage_pref_exp'].mean()


def comp_precision_seq(data):
    data['replaced_percentage_pref_pred'] = data.apply(
        lambda row: comp_fraction_prefix_seq(row['replaced_common_prefix_len'],
                                             row['replaced_filtered_beamsearch_ast_seq_len']),
        axis=1)

    data['percentage_pref_pred'] = data.apply(
        lambda row: comp_fraction_prefix_seq(row['common_prefix_len'],
                                             row['beamsearch_ast_seq_len']),
        axis=1)

    return data['replaced_percentage_pref_pred'].mean(), data['percentage_pref_pred'].mean()


def comp_fraction_prefix_seq(prefix, seq):
    return (prefix * 100) / seq


def comp_recall_token(data):
    replaced = (data['replaced_common_prefix_len'].sum() * 100) / data['replaced_filtered_expected_ast_seq_len'].sum()
    common = (data['common_prefix_len'].sum() * 100) / data['expected_ast_seq_len'].sum()
    return replaced, common


def comp_precision_token(data):
    replaced = (data['replaced_common_prefix_len'].sum() * 100) / data['replaced_filtered_beamsearch_ast_seq_len'].sum()
    common = (data['common_prefix_len'].sum() * 100) / data['beamsearch_ast_seq_len'].sum()
    return replaced, common


def comp_exact_match_acc(data):
    data['replaced_exact_match'] = data.apply(
        lambda row: comp_exact_match(row['replaced_common_prefix_len'], row['replaced_filtered_expected_ast_seq_len']),
        axis=1)

    data['exact_match'] = data.apply(
        lambda row: comp_exact_match(row['common_prefix_len'], row['expected_ast_seq_len']), axis=1)

    return data['replaced_exact_match'].mean() * 100, data['exact_match'].mean() * 100


def comp_exact_match(prefix_len, expected_len):
    return 1 if prefix_len == expected_len else 0


def compute_brevity_penalty(data):
    data['pred_len_less_than_exp_len'] = data.apply(
        lambda row: (row['beamsearch_ast_seq_len'] < row['expected_ast_seq_len']), axis=1)
    print('pred_len > exp_len: ',
          data.loc[~ data['pred_len_less_than_exp_len'], 'pred_len_less_than_exp_len'].count())
    print('pred_len < exp_len: ',
          data.loc[data['pred_len_less_than_exp_len'], 'pred_len_less_than_exp_len'].count())

    pred_len_sum = data['beamsearch_ast_seq_len'].sum()
    exp_len_sum = data['expected_ast_seq_len'].sum()
    brevity_penalty = math.exp(1 - 1 / min(1, (pred_len_sum / exp_len_sum)))
    print('Brevity Penalty: math.exp(1-1/min(1,(pred_len_sum/exp_len_sum))) = ', brevity_penalty)

    data['brevity_penalty_per_seq'] = data.apply(
        lambda row: min(1, row['beamsearch_ast_seq_len'] / row['expected_ast_seq_len']), axis=1)
    print('Average Brevity Penalty per SEQ (True: pred_len<exp_len )',
          data.groupby('pred_len_less_than_exp_len')['brevity_penalty_per_seq'].mean())


def compute_prefix_stats(data):
    # Recall: The fraction of the total amount of correctly predicted instances
    replaced_recall_seq, recall_seq = comp_recall_seq(data)
    print(
        'Replaced - The average length fraction of the correctly predicted prefix from the expected sequence (recall; seq len not relevant):',
        replaced_recall_seq, '%')
    print(
        'The average length fraction of the correctly predicted prefix from the expected sequence (recall; seq len not relevant):',
        recall_seq, '%')

    replaced_recall_token, recall_token = comp_recall_token(data)
    print(
        'Replaced - The fraction of tokens in correctly predicted prefixes from the tokens of expected sequences (recall; seq len relevant):',
        replaced_recall_token, '%')
    print(
        'The fraction of tokens in correctly predicted prefixes from the tokens of expected sequences (recall; seq len relevant):',
        recall_token, '%')

    # Precision: The fraction of correctly predicted instances among the predicted instances
    replaced_precision_seq, precision_seq = comp_precision_seq(data)
    print(
        'Replaced - The average length fraction in correctly predicted prefixes from the tokens of predicted sequences (precision; seq len relevant):',
        replaced_precision_seq, '%')
    print(
        'The average length fraction in correctly predicted prefixes from the tokens of predicted sequences (precision; seq len relevant):',
        precision_seq, '%')

    replaced_precision_token, precision_token = comp_precision_token(data)
    print(
        'Replaced - The fraction of tokens in correctly predicted prefixes from the tokens of predicted sequences (precision; seq len relevant):',
        replaced_precision_token, '%')
    print(
        'The fraction of tokens in correctly predicted prefixes from the tokens of predicted sequences (precision; seq len relevant):',
        precision_token, '%')

    # Exact match accuracy
    replaced_em, em = comp_exact_match_acc(data)
    print('Replaced - Exact match token accuracy:', replaced_em, '%')
    print('Exact match token accuracy:', em, '%')


def compute_probs_stats(data):
    data['score_difference'] = data.apply(
        lambda row: row['target_expected_scores_sum'] - row['target_predicted_scores_sum'], axis=1)
    # mean_difference = data['target_expected_scores_sum'].mean() - data['target_predicted_scores_sum'].mean()
    print("Stats of differences of the log probabilities of the expected and the predicted AST seq:")
    print(data['score_difference'].describe())


def process_row(row):
    diff_analysis = row['diff_analysis_res']
    ast_seq_exp_type = row['ast_seq_exp_type']
    ast_seq_pred_type = row['ast_seq_pred_type']

    row_exp_type_count = Counter(ast_seq_exp_type)
    row_pred_type_count = Counter(ast_seq_pred_type)

    row_exp_counters = {}
    row_pred_counters = {}

    for diff_type in ['equal', 'insert', 'delete', 'replace']:
        exp_counter = Counter()
        pred_counter = Counter()
        intervals = diff_analysis[diff_type] if diff_type in diff_analysis else []
        for interval in intervals:
            expected_types = interval['type_exp']
            exp_counter.update(expected_types)

            predicted_type = interval['type_pred']
            pred_counter.update(predicted_type)
        row_exp_counters[diff_type] = exp_counter
        row_pred_counters[diff_type] = pred_counter
    return {
        'expected': dict(row_exp_type_count),
        'predicted': dict(row_pred_type_count),
        'equal': dict(row_pred_counters['equal'])
    }


def divide_by_type(equal_types, reference_types):
    ratios = {}
    for reference_type, reference_count in reference_types.items():
        predicted_type_count = equal_types[reference_type] if reference_type in equal_types else 0
        ratios[reference_type] = predicted_type_count / reference_count
    return ratios


def add_diff_type_counters(dict_1, dict_2):
    return {
        'expected': dict(Counter(dict_1['expected']) + Counter(dict_2['expected'])),
        'predicted': dict(Counter(dict_1['predicted']) + Counter(dict_2['predicted'])),
        'equal': dict(Counter(dict_1['equal']) + Counter(dict_2['equal']))
    }


def compute_diff_stats(data):
    data['diff_type_counters'] = data.apply(lambda row: process_row(row), axis=1)
    data['diff_precision'] = data.apply(
        lambda row: divide_by_type(row['diff_type_counters']['equal'], row['diff_type_counters']['predicted']), axis=1)
    data['diff_recall'] = data.apply(
        lambda row: divide_by_type(row['diff_type_counters']['equal'], row['diff_type_counters']['expected']), axis=1)

    global_type_counters = reduce(lambda x, y: add_diff_type_counters(x, y), data['diff_type_counters'])
    global_precision = divide_by_type(global_type_counters['equal'], global_type_counters['predicted'])
    global_recall = divide_by_type(global_type_counters['equal'], global_type_counters['expected'])

    print("Global counts by type: %s" % global_type_counters)
    print("Global precision by type: %s" % global_precision)
    print("Global recall by type: %s" % global_recall)


def get_args():
    parser = argparse.ArgumentParser("Error analysis results", fromfile_prefix_chars='@')
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--compute-prefix-stats", action='store_true')
    parser.add_argument("--compute-probs-stats", action='store_true')
    parser.add_argument("--compute-diff-stats", action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    print("Error analysis args:", vars(args))

    data = load_json(args.input_file)

    if args.compute_prefix_stats:
        compute_prefix_stats(data)

    if args.compute_probs_stats:
        compute_probs_stats(data)

    if args.compute_diff_stats:
        compute_diff_stats(data)

    compute_brevity_penalty(data)


if __name__ == '__main__':
    main()
