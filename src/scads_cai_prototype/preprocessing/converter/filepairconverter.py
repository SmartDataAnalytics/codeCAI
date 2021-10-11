import argparse

import pandas as pd

from ..filehandling import save_json


class FilePairConverter:
    def __init__(self, src_input_path, tgt_input_path, output_path, newline_replacement, id_prefix):
        super(FilePairConverter, self).__init__()
        self.src_input_path = src_input_path
        self.tgt_input_path = tgt_input_path
        self.output_path = output_path
        self.newline_replacement = newline_replacement
        self.id_prefix = id_prefix

    def convert(self):
        src_lines = pd.read_csv(self.src_input_path, header=0, sep='\0', names=['intent'])
        tgt_lines = pd.read_csv(self.tgt_input_path, header=0, sep='\0', names=['snippet'])

        if (len(src_lines) != len(tgt_lines)):
            raise ValueError(
                "Source and target file have different number of lines: %s vs. %s lines" % (
                    len(src_lines), len(tgt_lines)))

        output_df = pd.DataFrame()
        output_df['intent'] = src_lines['intent']
        output_df['snippet'] = self.replace_nl(pd.Series(tgt_lines['snippet']))
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
    parser.add_argument("--src-input-path", type=str, required=True)
    parser.add_argument("--tgt-input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--newline-replacement", type=str, required=True)
    parser.add_argument("--id-prefix", type=str, default='')

    return parser.parse_args()


def main():
    args = get_args()
    print("Converter args:", vars(args))
    converter = FilePairConverter(src_input_path=args.src_input_path,
                                  tgt_input_path=args.tgt_input_path,
                                  output_path=args.output_path,
                                  newline_replacement=args.newline_replacement,
                                  id_prefix=args.id_prefix,
                                  )
    converter.convert()


if __name__ == '__main__':
    main()
