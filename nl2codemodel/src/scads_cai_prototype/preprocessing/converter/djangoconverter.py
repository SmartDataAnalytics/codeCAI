import argparse
import ast

import astor
import pandas as pd

from scads_cai_prototype.preprocessing.filehandling import load_json, save_json

'''
This script converts the Django dataset pre-processed by Yin and Neubig (2017) to the expected input format for the preproc.py script.
'''


# Please follow these steps:
#
# 1. Clone https://github.com/pcyin/NL2code
# 2. Setup the development environment and obtain a copy of pre-processed datasets by Yin and Neubig (2017) as described in the README.md of the NL2code repository
#    The code appears to expect Python 2.7 and additional dependencies not described in the README.md.
#    Anaconda 4.2.0 might be a good starting point: https://docs.anaconda.com/anaconda/packages/old-pkg-lists/4.2.0/py27/
# 3. Open the project PyCharm (or any Python IDE/debugger with Python console):
# 4. Create a breakpoint in code_gen.py line 128
# 5. Execute code_gen.py in debug mode with the following settings:
#    Working directory: The main directory of the NL2code working copy
#    Parameters: -data_type django -data data/django.cleaned.dataset.freq5.par_info.refact.space_only.bin -output_dir runs -model models/model.django_word128_encoder256_rule128_node64.beam15.adam.simple_trans.no_unary_closure.8e39832.run3.best_acc.npz -rule_embed_dim 128 -node_embed_dim 64 decode -saveto runs/model.django_word128_encoder256_rule128_node64.beam15.adam.simple_trans.no_unary_closure.8e39832.run3.best_acc.npz.decode_results.test.bin
#    Environment variables: PYTHONUNBUFFERED=1;THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32
#
# 6. When breakpoint hits, execute the following commands in console:
#    import pandas as pd
#    def convert(dataset):
#      return pd.DataFrame({
#          'code': ex.code,
#          'query': ex.query,
#          'raw_id':ex.raw_id,
#          'replacements': {name: value for value, name in ex.meta_data['str_map'].iteritems()}
#        } for ex in dataset.examples)
#    dev_df = convert(dev_data)
#    train_df = convert(train_data)
#    test_df = convert(test_data)
#
#    dev_df.to_json('django-YN17-dev.jsonl', orient='records', lines=True)
#    train_df.to_json('django-YN17-train.jsonl', orient='records', lines=True)
#    test_df.to_json('django-YN17-test.jsonl', orient='records', lines=True)
#
# 7. Use the files django-YN17-dev.jsonl, django-YN17-train.jsonl and django-YN17-test.jsonl as input for this program

class DjangoConverter:
    def __init__(self, input_path, output_path, id_prefix, apply_replacements):
        self.input_path = input_path
        self.output_path = output_path
        self.id_prefix = id_prefix
        self.apply_replacements = apply_replacements

    def convert(self):
        input_df = load_json(self.input_path)
        output_df = pd.DataFrame()

        output_df['intent'] = input_df.apply(lambda row: self.convert_query(row['query'], row['replacements']), axis=1)
        output_df['snippet'] = input_df.apply(lambda row: self.convert_code(row['code'], row['replacements']), axis=1)
        output_df['id'] = input_df['raw_id'].apply(lambda raw_id: self.convert_id(raw_id))
        output_df['django_line_number'] = input_df['raw_id']
        if not self.apply_replacements:
            # Needed for post-processing predictions (not for training)
            output_df['replacements'] = input_df['replacements']
        save_json(output_df, self.output_path)

    def convert_query(self, query, replacements):
        if self.apply_replacements:
            query = (replacements[token] if token in replacements else token for token in query)

        return ' '.join(query)

    def convert_code(self, code, replacements):
        if self.apply_replacements:
            for name, value in replacements.items():
                code = code.replace("'" + name + "'", value)

        return astor.to_source(ast.parse(code))

    def convert_id(self, raw_id):
        return self.id_prefix + str(raw_id)


def get_args():
    parser = argparse.ArgumentParser("Convert text file pairs to JSON", fromfile_prefix_chars='@')

    # Dataset parameters
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--id-prefix", type=str, default='')
    parser.add_argument("--apply-replacements", action='store_true')

    return parser.parse_args()


def main():
    args = get_args()
    print("Converter args:", vars(args))
    converter = DjangoConverter(input_path=args.input_path,
                                output_path=args.output_path,
                                id_prefix=args.id_prefix,
                                apply_replacements=args.apply_replacements)
    converter.convert()


if __name__ == '__main__':
    main()
