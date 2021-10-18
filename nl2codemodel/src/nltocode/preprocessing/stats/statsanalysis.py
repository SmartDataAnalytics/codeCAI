import argparse
import json

import pandas as pd


class StatsAnalysis():

    def __init__(self,
                 record_stats_input_file,
                 global_stats_input_file,
                 train_data_input_file,
                 data_analysis_stats_output_file,
                 list_len_threshold,
                 lit_len_threshold,
                 lit_freq_threshold,
                 prob_threshold,
                 filter_duplicate_ast_seq
                 ):
        self.record_stats_input_file = record_stats_input_file
        self.global_stats_input_file = global_stats_input_file
        self.train_data_input_file = train_data_input_file
        self.data_analysis_stats_output_file = data_analysis_stats_output_file

        self.list_len_threshold = list_len_threshold
        self.lit_len_threshold = lit_len_threshold
        self.lit_freq_threshold = lit_freq_threshold
        self.prob_threshold = prob_threshold
        self.filter_duplicate_ast_seq = filter_duplicate_ast_seq

    def analyze_record_stats(self):
        to_be_deleted = set()

        if self.prob_threshold is not None:
            train_data_df = self.load_data(self.train_data_input_file)
            print('to_be_deleted_by_prob_th')
            to_be_deleted.update(self.filter_snippets_by_prob_threshold(train_data_df))

        record_stats_df = self.load_data(self.record_stats_input_file)

        if self.lit_len_threshold is not None:
            print('to_be_deleted_by_lit_length_th')
            to_be_deleted.update(self.filter_snippets_by_lit_length_threshold(record_stats_df))

        if self.list_len_threshold is not None:
            print('to_be_deleted_by_list_length_th')
            to_be_deleted.update(self.filter_snippets_by_list_length_threshold(record_stats_df))

        if self.filter_duplicate_ast_seq:
            print('to_be_deleted_by_duplicate_ast_seq')
            to_be_deleted.update(self.filter_snippets_by_duplicate_ast_seq(record_stats_df))

        if self.lit_freq_threshold is not None:
            print('to_be_deleted_by_lit_freq_threshold')
            to_be_deleted.update(self.filter_snippets_by_lit_freq_threshold(record_stats_df))

        self.save_data(sorted(str(id) + "\n" for id in to_be_deleted), self.data_analysis_stats_output_file)

    def filter_snippets_by_duplicate_ast_seq(self, record_stats_df):
        record_stats_df['ast_seq_list_tuple'] = record_stats_df['ast_seq'].apply(lambda ast_seq: tuple(ast_seq))

        record_stats_df['to_be_deleted_by_duplicate_ast_seq'] = record_stats_df.duplicated('ast_seq_list_tuple')

        to_be_deleted_by_duplicate_ast_seq = record_stats_df[record_stats_df['to_be_deleted_by_duplicate_ast_seq']]

        for l in to_be_deleted_by_duplicate_ast_seq['id']:
            print(l)
        return set(to_be_deleted_by_duplicate_ast_seq['id'])

    def filter_snippets_by_prob_threshold(self, train_data_df):
        train_data_df['to_be_deleted_by_prob_th'] = train_data_df['prob'] < self.prob_threshold

        to_be_deleted_by_prob_th = train_data_df[train_data_df['to_be_deleted_by_prob_th']]

        for l in to_be_deleted_by_prob_th['id']:
            print(l)
        return set(to_be_deleted_by_prob_th['id'])

    def filter_snippets_by_list_length_threshold(self, record_stats_df):
        record_stats_df['to_be_deleted_by_list_len_th'] = record_stats_df.apply(
            lambda row: self.get_snippets_by_list_length_threshold(row['list_len_histogram']),
            axis=1)

        to_be_deleted_by_list_len_th = record_stats_df[record_stats_df['to_be_deleted_by_list_len_th']]
        # print('to_be_deleted_by_list_len_th', to_be_deleted_by_list_len_th['id'].tolist())

        for l in to_be_deleted_by_list_len_th['id']:
            print(l)
        return set(to_be_deleted_by_list_len_th['id'])

    def get_snippets_by_list_length_threshold(self, list_len_histogram):
        for list_len in list_len_histogram:
            if int(list_len) > self.list_len_threshold:
                return True

        return False

    def filter_snippets_by_lit_length_threshold(self, record_stats_df):
        record_stats_df['to_be_deleted_by_len_th'] = record_stats_df.apply(
            lambda row: self.get_snippets_by_lit_length_threshold(row['literal_histogram']),
            axis=1)

        to_be_deleted_by_len_th = record_stats_df[record_stats_df['to_be_deleted_by_len_th']]
        # print('to_be_deleted_by_len_th', to_be_deleted_by_len_th['id'].tolist())
        for l in to_be_deleted_by_len_th['id']:
            print(l)
        return set(to_be_deleted_by_len_th['id'])

    def get_snippets_by_lit_length_threshold(self, literal_histogram):
        for literal in literal_histogram:
            if len(literal) > self.lit_len_threshold:
                return True

        return False

    def filter_snippets_by_lit_freq_threshold(self, record_stats_df):
        global_stats = self.load_json(self.global_stats_input_file)

        filtered_global_literal_histogram = self.filter_histogram(global_stats['global_literal_histogram'],
                                                                  0, self.lit_freq_threshold)

        record_stats_df['to_be_deleted_by_freq_th'] = record_stats_df.apply(
            lambda row: self.get_snippets_by_lit_freq_threshold(row['literal_histogram'],
                                                                filtered_global_literal_histogram), axis=1)

        to_be_deleted_by_freq_th = record_stats_df[record_stats_df['to_be_deleted_by_freq_th']]
        # print('to_be_deleted_by_freq_th', to_be_deleted_by_freq_th['id'].tolist())
        for l in to_be_deleted_by_freq_th['id']:
            print(l)
        return set(to_be_deleted_by_freq_th['id'])

    def get_snippets_by_lit_freq_threshold(self, lit_hist, filtered_global_lit_hist):
        return any(k in filtered_global_lit_hist for k in lit_hist)

    def filter_histogram(self, d, min_value, max_value):
        return {k: v for k, v in d.items() if min_value <= v <= max_value}

    def load_data(self, path):
        is_jsonl = path.lower().endswith('.jsonl')
        return pd.read_json(path, lines=is_jsonl, dtype=False)

    def load_json(self, path):
        with open(path) as json_file:
            return json.load(json_file)

    def save_data(self, data, filename):
        with open(filename, 'w') as f:
            f.writelines(data)


def get_args():
    parser = argparse.ArgumentParser("Analyze record stats", fromfile_prefix_chars='@')

    parser.add_argument("--record-stats-input-file", type=str, default="record-stats.jsonl")
    parser.add_argument("--global-stats-input-file", type=str, default="global-stats.json")
    parser.add_argument("--train-data-input-file", type=str, default="conala-mined.jsonl")
    parser.add_argument("--list-len-threshold", type=int, default=None)
    parser.add_argument("--lit-len-threshold", type=int, default=None)
    parser.add_argument("--lit-freq-threshold", type=int, default=None)
    parser.add_argument("--prob-threshold", type=float, default=None)
    parser.add_argument("--data-record-analysis-stats-output-file", type=str, default="data-stats-analysis-res.csv")
    parser.add_argument("--filter-duplicate-ast-seq", type=bool_str, default=False)

    return parser.parse_args()


def bool_str(val):
    if val == 'True':
        return True
    elif val == 'False':
        return False
    else:
        raise ValueError('Unexpected bool value: ', val)


def main():
    args = get_args()
    print("Stats analysis args:", vars(args))

    stats = StatsAnalysis(
        record_stats_input_file=args.record_stats_input_file,
        global_stats_input_file=args.global_stats_input_file,
        train_data_input_file=args.train_data_input_file,
        list_len_threshold=args.list_len_threshold,
        lit_len_threshold=args.lit_len_threshold,
        lit_freq_threshold=args.lit_freq_threshold,
        prob_threshold=args.prob_threshold,
        data_analysis_stats_output_file=args.data_record_analysis_stats_output_file,
        filter_duplicate_ast_seq=args.filter_duplicate_ast_seq
    )

    stats.analyze_record_stats()


if __name__ == '__main__':
    main()
