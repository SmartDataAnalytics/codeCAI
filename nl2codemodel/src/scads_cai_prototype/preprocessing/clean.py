import argparse

import pandas as pd


class Clean():

    def __init__(self,
                 data_input_file,
                 data_output_file,
                 blacklist_file
                 ):
        self.data_input_file = data_input_file
        self.data_output_file = data_output_file
        self.blacklist_file = blacklist_file

    def curate(self):
        in_df = self.load_data(self.data_input_file)
        if 'id' not in in_df.columns:
            raise ValueError("No column named 'id' in input file", self.data_output_file)
        blacklist = self.load_blacklist(self.blacklist_file)

        # blacklist can only contain strings, so the id column has to be converted to string
        to_be_retained = in_df['id'].apply(lambda id: (str(id) not in blacklist))
        out_df = in_df[to_be_retained]
        self.save_data(out_df, self.data_output_file)

    def load_data(self, path):
        is_jsonl = path.lower().endswith('.jsonl')
        return pd.read_json(path, lines=is_jsonl, dtype=False)

    def load_blacklist(self, path):
        with open(path) as f:
            return set(line.rstrip() for line in f)

    def save_data(self, dataframe, filename):
        is_jsonl = filename.endswith('.jsonl')
        dataframe.to_json(filename, orient='records', lines=is_jsonl)


def get_args():
    parser = argparse.ArgumentParser("Curate data", fromfile_prefix_chars='@')

    parser.add_argument("--data-input-file", type=str, default="input-data.jsonl")
    parser.add_argument("--data-output-file", type=str, default="output-data.jsonl")
    parser.add_argument("--blacklist-file", type=str, default="blacklist.csv")

    return parser.parse_args()


def main():
    args = get_args()
    print("Clean args:", vars(args))

    stats = Clean(
        data_input_file=args.data_input_file,
        data_output_file=args.data_output_file,
        blacklist_file=args.blacklist_file
    )

    stats.curate()


if __name__ == '__main__':
    main()
