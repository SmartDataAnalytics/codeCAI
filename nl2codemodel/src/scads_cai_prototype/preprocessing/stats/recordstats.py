import pandas as pd

import argparse

from collections import Counter
from collections import defaultdict

class RecordStats():

    def __init__(self,
                 preproc_data_stats_input_file,
                 stats_data_output_file
                 ):

        self.preproc_data_stats_input_file = preproc_data_stats_input_file
        self.stats_data_output_file = stats_data_output_file


    def compute_stats(self):
        stats_df = self.load_data(self.preproc_data_stats_input_file)

        #print(stats_df.to_string())

        stats_df[['max_edge_count', 'min_max_list_el', 'list_len_histogram', 'node_histogram', 'edge_histogram',
                  'literal_histogram', 'max_lit_lengths', 'node_count', 'list_count', 'literal_count', 'edge_count']] \
            = stats_df['edge_counts'].apply(lambda ecnt: pd.Series(self.comp_edge_counts_stats(ecnt)))

        self.save_data(stats_df, self.stats_data_output_file)

    def comp_edge_counts_stats(self, edge_counts):
        max_edge_count = self.comp_max_edge_count(edge_counts)
        min_max_list_el = self.comp_min_max_list_el(edge_counts)
        list_len_histogram = self.comp_list_len_histogram(edge_counts)
        node_histogram = self.comp_node_histogram(edge_counts)
        edge_histogram = self.comp_edge_histogram(edge_counts)
        literal_histogram = self.comp_literal_histogram(edge_counts)
        max_lit_lengths = self.comp_max_lit_lengths(literal_histogram)
        node_count = self.comp_node_count(edge_counts)
        list_count = self.comp_list_count(edge_counts)
        literal_count = self.comp_literal_count(edge_counts)
        edge_count = self.comp_edge_count(edge_histogram)

        return max_edge_count, min_max_list_el, list_len_histogram, node_histogram, edge_histogram, literal_histogram, max_lit_lengths, node_count, list_count, literal_count, edge_count

    def comp_max_edge_count(self, edge_counts):
        max_edge_count = 0

        for item in edge_counts:
            edge_count = item[2]
            if max_edge_count < edge_count:
                max_edge_count = edge_count

        return max_edge_count

    def comp_min_max_list_el(self, edge_counts):
        min_max_list_el = {}
        edge_dict = defaultdict(list)

        for label, type_, count in edge_counts:
            if type_ == 'list':
                edge_dict[label].append(count)

        for key, value in edge_dict.items():
            sorted_by_count = sorted(value)
            min_max_list_el[key] = (sorted_by_count[0], sorted_by_count[-1])

        return min_max_list_el

    def comp_list_len_histogram(self, edge_counts):
        return self.comp_counts(list=(l for l in edge_counts if l[1] == 'list'), idx=2)

    def comp_node_histogram(self, edge_counts):
        return self.comp_counts(list=(l for l in edge_counts if l[1] == 'node'), idx=0)

    def comp_edge_histogram(self, edge_counts):
        edge_cnt = Counter()
        for item in edge_counts:
            if item[1] == 'list' or item[1] == 'singleton':
                edge_cnt[item[0]] += item[2]

        return edge_cnt

    def comp_literal_histogram(self, edge_counts):
        get_literal_repr = lambda literal: literal[:literal.rfind('#')]

        return self.comp_counts(list=([get_literal_repr(n), t, c] for n, t, c in edge_counts if t in ('literal', 'strliteral')), idx=0)

    def comp_max_lit_lengths(self, literal_histogram):
        max_len = 0
        for literal in literal_histogram:
            if max_len < len(literal):
                max_len = len(literal)

        return max_len

    def comp_node_count(self, edge_counts):
        return self.comp_count_by_type(edge_counts, ['node'])

    def comp_list_count(self, edge_counts):
        return self.comp_count_by_type(edge_counts, ['list'])

    def comp_literal_count(self, edge_counts):
        return self.comp_count_by_type(edge_counts, ['literal', 'strliteral'])

    def comp_count_by_type(self, edge_counts, types):
        return sum(1 for n, t, c in edge_counts if t in types)

    def comp_edge_count(self, edge_histogram):
        return sum(edge_histogram.values())

    def comp_counts(self, list, idx):
        cnt = Counter()
        for item in list:
            cnt[item[idx]] += 1

        return cnt

    def load_data(self, path):
        is_jsonl = path.lower().endswith('.jsonl')
        return pd.read_json(path, lines=is_jsonl, dtype=False)

    def save_data(self, dataframe, filename):
        is_jsonl = filename.endswith('.jsonl')
        dataframe.to_json(filename, orient='records', lines=is_jsonl)


def get_args():
    parser = argparse.ArgumentParser("Preproc data", fromfile_prefix_chars='@')

    parser.add_argument("--preproc-data-stats-input-file", type=str, default="preproc-data-stats.jsonl")
    parser.add_argument("--stats-data-output-file", type=str, default="stats-data.jsonl")

    return parser.parse_args()


def main():
    args = get_args()
    print("Record stats args:", vars(args))

    stats = RecordStats(
        preproc_data_stats_input_file=args.preproc_data_stats_input_file,
        stats_data_output_file=args.stats_data_output_file
    )

    stats.compute_stats()


if __name__ == '__main__':
    main()
