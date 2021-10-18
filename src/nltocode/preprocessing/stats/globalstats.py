import argparse
import json

from collections import Counter
from collections import defaultdict

import pandas as pd
class GlobalStats():

    def __init__(self,
                 data_stats_input_file,
                 data_global_stats_output_file
                 ):

        self.data_stats_input_file = data_stats_input_file
        self.data_global_stats_output_file = data_global_stats_output_file

    def compute_global_stats(self):
        stats_df = self.load_data(self.data_stats_input_file)

        global_stats = {}

        global_stats['global_max_depth'] = self.comp_global_max_depth(stats_df['max_depth'].tolist())
        global_stats['global_max_edge_count'] = self.comp_global_max_edge_count(stats_df['max_edge_count'].tolist())
        global_stats['global_min_max_list_el'] = self.comp_global_min_max_list_el(stats_df['min_max_list_el'].tolist())
        global_stats['global_list_len_histogram'] = self.comp_global_list_len_histogram(
            stats_df['list_len_histogram'].tolist())
        global_stats['global_node_histogram'] = self.comp_global_node_histogram(stats_df['node_histogram'].tolist())
        global_stats['global_edge_histogram'] = self.comp_global_edge_histogram(stats_df['edge_histogram'].tolist())
        global_stats['global_literal_histogram'] = self.comp_global_literal_histogram(
            stats_df['literal_histogram'].tolist())
        global_stats['max_lit_len_histogram'] = self.comp_max_lit_len_histogram(stats_df['max_lit_lengths'].tolist())
        global_stats['min_lit_freq_histogram'] = self.comp_min_lit_freq_histogram(stats_df['literal_histogram'].tolist(), global_stats['global_literal_histogram'], stats_df['id'].tolist())
        global_stats['global_node_count'] = self.comp_global_node_count(stats_df['node_count'].tolist())
        global_stats['global_list_count'] = self.comp_global_list_count(stats_df['list_count'].tolist())
        global_stats['global_literal_count'] = self.comp_global_literal_count(stats_df['literal_count'].tolist())
        global_stats['global_distinct_literal_count'] = self.comp_global_distinct_literal_count(
            global_stats['global_literal_histogram'])
        global_stats['global_distinct_node_count'] = self.comp_global_distinct_literal_count(
            global_stats['global_node_histogram'])
        global_stats['global_edge_count'] = self.comp_global_edge_count(stats_df['edge_count'].tolist())

        # print(global_stats)
        self.save_data(global_stats, self.data_global_stats_output_file)

    def comp_global_max_depth(self, max_depth):
        return max(max_depth)

    def comp_global_max_edge_count(self, max_edge_count):
        return max(max_edge_count)

    def comp_global_min_max_list_el(self, min_max_list_el):
        global_min_max_list_el = {}
        global_edge_dict = defaultdict(list)
        for el in min_max_list_el:
            for key, value in el.items():
                global_edge_dict[key].extend(value)
        for key, value in global_edge_dict.items():
            sorted_by_count = sorted(value)
            global_min_max_list_el[key] = {'min': sorted_by_count[0], 'max': sorted_by_count[-1]}

        return self.sort_dict_by_key(global_min_max_list_el)

    def comp_global_list_len_histogram(self, list_len_histogram):
        return self.sort_dict_by_key(self.comp_global_counts(list_len_histogram))

    def comp_global_node_histogram(self, node_histogram):
        return self.sort_dict_by_value(self.comp_global_counts(node_histogram))

    def comp_global_edge_histogram(self, edge_histogram):
        return self.sort_dict_by_value(self.comp_global_counts(edge_histogram))

    def comp_global_literal_histogram(self, literal_histogram):
        return self.sort_dict_by_value(self.comp_global_counts(literal_histogram))

    def comp_max_lit_len_histogram(self, max_lit_lengths):
        return self.sort_dict_by_key(self.comp_counts(list=max_lit_lengths))

    def comp_min_lit_freq_histogram(self, literal_histograms, global_lit_histogram, snippet_ids):
        min_global_lit_freq = []
        for literal_histogram, snippet_id in zip(literal_histograms, snippet_ids):
            min_global_lit_freq.append(self.comp_min_lit_frequency(literal_histogram, global_lit_histogram, snippet_id))

        return self.sort_dict_by_key(self.comp_counts(list=min_global_lit_freq))

    def comp_min_lit_frequency(self, literal_histogram, global_lit_histogram, snippet_id):
        global_freq = []
        for literal in literal_histogram:
            global_freq.append(global_lit_histogram[literal])

        if global_freq:
            return min(global_freq)
        else:
            print("No literals in snippet ", snippet_id)
            return -1

    def comp_global_node_count(self, node_count):
        return sum(node_count)

    def comp_global_list_count(self, list_count):
        return sum(list_count)

    def comp_global_literal_count(self, literal_count):
        return sum(literal_count)

    def comp_global_distinct_node_count(self, global_node_histogram):
        return len(global_node_histogram)

    def comp_global_distinct_literal_count(self, global_literal_histogram):
        return len(global_literal_histogram)

    def comp_global_edge_count(self, edge_count):
        return sum(edge_count)

    def comp_global_counts(self, items_list):
        cnt = Counter()
        for item in items_list:
            cnt.update(item)

        return dict(cnt)

    def comp_counts(self, list):
        cnt = Counter()
        for item in list:
            cnt[item] += 1

        return dict(cnt)

    def sort_dict_by_key(self, d):
        return {k: v for k, v in sorted(d.items(), key=lambda e: e[0])}

    def sort_dict_by_value(self, d):
        return {k: v for k, v in sorted(d.items(), key=lambda e: e[1])}

    def load_data(self, path):
        is_jsonl = path.lower().endswith('.jsonl')
        return pd.read_json(path, lines=is_jsonl, dtype=False)

    def save_data(self, data, filename):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


def get_args():
    parser = argparse.ArgumentParser("Global dataset stats", fromfile_prefix_chars='@')

    parser.add_argument("--data-stats-input-file", type=str, default="data-stats.jsonl")
    parser.add_argument("--data-global-stats-output-file", type=str, default="global-data-stats.jsonl")

    return parser.parse_args()


def main():
    args = get_args()
    print("Global stats args:", vars(args))

    stats = GlobalStats(
        data_stats_input_file=args.data_stats_input_file,
        data_global_stats_output_file=args.data_global_stats_output_file
    )

    stats.compute_global_stats()


if __name__ == '__main__':
    main()
