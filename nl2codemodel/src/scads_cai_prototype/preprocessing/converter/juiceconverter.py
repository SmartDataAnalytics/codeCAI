import argparse

import pandas as pd


class JuICeConverter:
    def __init__(self, input_data_path, output_data_path):
        self.input_data_path = input_data_path
        self.output_data_path = output_data_path

    def convert_data(self) -> pd.DataFrame:
        print("Loading from", self.input_data_path)
        juice_df = self.load_data(self.input_data_path)
        print("Converting to nl-code pairs")
        juice_df['nl_code_pairs'] = juice_df.apply(lambda row: self.convert_juice_nb_cells(row['context'], row['code']),
                                                   axis=1)
        print("Concatenating")
        all_nl_code_pairs_with_id = self.concatenate_all_nl_code_pairs(juice_df['nl_code_pairs'])
        print("Creating output dataframe")
        all_nl_code_pairs_df = pd.DataFrame(all_nl_code_pairs_with_id, columns=['id', 'intent', 'snippet'])
        print("Saving to", self.output_data_path)
        self.save_data(all_nl_code_pairs_df, self.output_data_path)

    def concatenate_all_nl_code_pairs(self, nl_code_pairs):
        for nb_id, nl_code_pairs in enumerate(nl_code_pairs):
            for pair_id, (nl, code) in enumerate(nl_code_pairs):
                idx = str(nb_id + 1) + "_" + str(pair_id + 1)
                yield idx, nl, code

    def convert_juice_nb_cells(self, context, target_code):
        reversed_nb_cells = reversed(context)
        # print(list(reversed_nb_cells))

        nl_code_pairs = []
        iterator = iter(reversed_nb_cells)
        try:
            item = next(iterator)
            if item['cell_type'] not in ("markdown", "code"):
                print("Irregular cell type", item['cell_type'])
        except StopIteration:
            return []

        while True:
            md_list = []
            code_list = []
            try:
                while item['cell_type'] != "code":
                    if item['cell_type'] == "markdown":
                        md_list.append(item['nl_original'])
                    item = next(iterator)
                    if item['cell_type'] not in ("markdown", "code"):
                        print("Irregular cell type", item['cell_type'])
                while item['cell_type'] != "markdown":
                    if item['cell_type'] == "code":
                        code_list.append(item['code'])
                    item = next(iterator)
                    if item['cell_type'] not in ("markdown", "code"):
                        print("Irregular cell type", item['cell_type'])
            except StopIteration:
                code_list.append(target_code)
                self.append_nl_code_pair(code_list, md_list, nl_code_pairs)
                break
            self.append_nl_code_pair(code_list, md_list, nl_code_pairs)

        return nl_code_pairs

    def append_nl_code_pair(self, code_list, md_list, nl_code_pairs):
        nl = "\n".join(md_list)
        code = "\n".join(code_list)

        if nl and code:
            nl_code_pairs.append((nl, code))

    def to_nl_code_pair(self, md_list, code_list):
        nl = "\n".join(md_list)
        code = "\n".join(code_list)
        return (nl, code)

    def load_data(self, path) -> pd.DataFrame:
        is_jsonl = path.lower().endswith('.jsonl')
        return pd.read_json(path, lines=is_jsonl, dtype=False)


    def save_data(self, dataframe, filename):
        is_jsonl = filename.endswith('.jsonl')
        dataframe.to_json(filename, orient='records', lines=is_jsonl)


def get_args():
    parser = argparse.ArgumentParser("Analyze record stats", fromfile_prefix_chars='@')

    parser.add_argument("--juice-input-file", type=str, default="juice.jsonl")
    parser.add_argument("--juice-output-file", type=str, default="juice-converted.jsonl")

    return parser.parse_args()


def main():
    args = get_args()
    print("JuICe converter args:", vars(args))

    juice_data_provider = JuICeConverter(
        input_data_path=args.juice_input_file,
        output_data_path=args.juice_output_file
    )
    juice_data_provider.convert_data()


if __name__ == '__main__':
    main()
