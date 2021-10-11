import argparse

import pandas as pd

from scads_cai_prototype.preprocessing.filehandling import save_json


class CSVFileConverter:
    def __init__(self, input_path, output_path, separator, src_col, tgt_col, newline_replacement, id_prefix):
        super(CSVFileConverter, self).__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.separator = separator
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.newline_replacement = newline_replacement
        self.id_prefix = id_prefix

    def convert(self):
        columns = [self.src_col, self.tgt_col]
        names = ['intent', 'snippet'] if self.src_col < self.tgt_col else ['snippet', 'intent']
        data = pd.read_csv(self.input_path, sep=self.separator, usecols=columns, names=names)

        output_df = pd.DataFrame()
        output_df['intent'] = data['intent']
        output_df['snippet'] = self.replace_nl(pd.Series(data['snippet']))
        output_df['id'] = pd.Series(output_df.reset_index().index).apply(lambda idx: self.id_prefix + str(idx + 1))

        save_json(output_df, self.output_path)

    def replace_nl(self, intents):
        if self.newline_replacement is None:
            return intents
        else:
            return intents.apply(lambda snippet: snippet.replace(self.newline_replacement, '\n'))


def get_args():
    parser = argparse.ArgumentParser("Convert text file pairs to JSON", fromfile_prefix_chars='@')

    # Dataset parameters
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--separator", type=str, default=';')
    parser.add_argument("--src-col", type=int, default=0)
    parser.add_argument("--tgt-col", type=int, default=1)
    parser.add_argument("--newline-replacement", type=str, default='\\n')
    parser.add_argument("--id-prefix", type=str, default='')

    return parser.parse_args()


def main():
    args = get_args()
    print("Converter args:", vars(args))
    converter = CSVFileConverter(input_path=args.input_path,
                                 output_path=args.output_path,
                                 separator=args.separator,
                                 src_col=args.src_col,
                                 tgt_col=args.tgt_col,
                                 newline_replacement=args.newline_replacement,
                                 id_prefix=args.id_prefix,
                                 )
    converter.convert()


if __name__ == '__main__':
    main()
