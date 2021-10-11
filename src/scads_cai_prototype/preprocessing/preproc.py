import argparse
import ast
import itertools
import logging as log
import os
import sys
import textwrap
import traceback

import astor
import pandas as pd

from asdl.asdl import ASDLGrammar
from scads_cai_prototype.generator.codegenerator import PythonCodeGenerator
from scads_cai_prototype.grammar.grammargraph import PythonGrammarGraphCreator, AsdlGrammarGraphCreator
from scads_cai_prototype.grammar.grammargraphloader import GrammarGraphLoader
from scads_cai_prototype.grammar.grammargraphvisitor import GrammarGraphVisitor
from scads_cai_prototype.grammar.vocabulary import VocabularyLoader, VocabularyCreator
from scads_cai_prototype.preprocessing.filehandling import save_json
from scads_cai_prototype.preprocessing.preprocdataprovider import PreprocTrainPythonDataProvider, \
    PreprocTrainLambdaDCSDataProvider, PreprocTestLambdaDCSDataProvider, PreprocTestPythonDataProvider


class Preproc():
    def __init__(self,
                 train_valid_dataprovider,
                 test_dataprovider,
                 preproc_train_valid_data_output_file,
                 preproc_test_data_output_file,
                 preproc_train_valid_data_stats_output_file,
                 vocabsrc_file,
                 vocabsrc_type,
                 vocabtgt_file,
                 vocabsrc_size,
                 vocabchar_file,
                 grammar_graph_file,
                 process_train_data,
                 process_test_data,
                 all_strliteral_edges,
                 language,
                 lambdadcsgrammar,
                 eliminate_train_duplicates,
                 validate_ast_seq_enc=True,
                 vocabsrc_max_sentencepiece_length=16,
                 vocabsrc_split_by_whitespace=True,
                 vocabsrc_split_digits=False,
                 vocabsrc_treat_whitespace_as_suffix=False,
                 vocabsrc_byte_fallback=False,
                 vocabsrc_add_dummy_prefix=True,
                 vocabsrc_user_defined_symbols=''
                 ):

        self.train_valid_dataprovider = train_valid_dataprovider
        self.test_dataprovider = test_dataprovider
        self.preproc_train_valid_data_stats_output_file = preproc_train_valid_data_stats_output_file

        self.preproc_train_valid_data_output_file = preproc_train_valid_data_output_file
        self.preproc_test_data_output_file = preproc_test_data_output_file

        self.vocabsrc_file = vocabsrc_file
        self.vocabtgt_file = vocabtgt_file

        self.vocabsrc_size = vocabsrc_size
        self.vocabtgt_size = None

        self.vocabsrc_type = vocabsrc_type

        self.vocabchar_file = vocabchar_file

        self.grammar_graph_file = grammar_graph_file

        self.process_train_data = process_train_data
        self.process_test_data = process_test_data
        self.all_strliteral_edges = all_strliteral_edges

        self.language = language
        self.lambdadcsgrammar = lambdadcsgrammar
        self.eliminate_train_duplicates = eliminate_train_duplicates
        self.validate_ast_seq_enc = validate_ast_seq_enc

        self.vocabsrc_max_sentencepiece_length = vocabsrc_max_sentencepiece_length
        self.vocabsrc_split_by_whitespace = vocabsrc_split_by_whitespace
        self.vocabsrc_split_digits = vocabsrc_split_digits
        self.vocabsrc_treat_whitespace_as_suffix = vocabsrc_treat_whitespace_as_suffix
        self.vocabsrc_byte_fallback = vocabsrc_byte_fallback
        self.vocabsrc_add_dummy_prefix = vocabsrc_add_dummy_prefix
        self.vocabsrc_user_defined_symbols = vocabsrc_user_defined_symbols

    def prepare_data(self):
        vocab_src, vocab_char = self.create_or_load_src_and_char_vocab()
        vocab_tgt = self.create_or_load_tgt_vocab()
        created_grammar_graph = self.create_grammar_graph(vocab_tgt, vocab_src)

        if self.process_train_data:
            grammar_graph = created_grammar_graph or GrammarGraphLoader(self.grammar_graph_file).load_graph()
            self.preproc_train_valid_data(vocab_src, vocab_tgt, vocab_char, grammar_graph)

        if self.process_test_data:
            self.preproc_test_data(vocab_src, vocab_char)

    def preproc_train_valid_data(self, vocab_src, vocab_tgt, vocab_char, grammar_graph):
        print()
        print(textwrap.dedent("""\
                    ========================
                    Preprocessing Train Data
                    ========================
                """))
        train_valid_data = self.train_valid_dataprovider.provide_data(columns=['nl', 'ast_seq_list', 'ast', 'snippet',
                                                                               'id', 'ast_seq'])
        train_valid_data['nl_enc'] = self.encode_nl(train_valid_data['nl'], vocab_src)
        train_valid_data['nl_char_enc'] = self.encode_nl_char(train_valid_data['nl'], vocab_src, vocab_char)

        graphvisitor = GrammarGraphVisitor(grammar_graph)

        if self.language == 'python':
            codegenerator = PythonCodeGenerator(grammar_graph)

            train_valid_data['ast_seq_enc'] = train_valid_data.apply(lambda row:
                                                                     self.encode_ast_seq(row['ast_seq_list'],
                                                                                         row['snippet'],
                                                                                         row['id'],
                                                                                         grammar_graph,
                                                                                         vocab_src,
                                                                                         vocab_tgt), axis=1)

            if self.validate_ast_seq_enc:
                train_valid_data['ast_seq_enc'] = train_valid_data.apply(lambda row:
                                                                         self.validate_python_ast_seq(codegenerator,
                                                                                                      row[
                                                                                                          'ast_seq_enc'],
                                                                                                      row['ast'],
                                                                                                      row['snippet'],
                                                                                                      row['id']),
                                                                         axis=1)

        elif self.language == 'lambdadcs':
            train_valid_data['ast_seq_enc'] = train_valid_data.apply(lambda row:
                                                                     self.encode_ast_seq(row['ast_seq_list'],
                                                                                         row['snippet'],
                                                                                         row['id'],
                                                                                         grammar_graph,
                                                                                         vocab_src,
                                                                                         vocab_tgt
                                                                                         ),
                                                                     axis=1)

        train_valid_data.dropna(inplace=True)

        train_valid_data[['allowed_tokens', 'edge_order_seq', 'edge_counts', 'max_depth']] = train_valid_data.apply(
            lambda row: pd.Series(
                graphvisitor.visit_graph_edge_order_path(row['ast_seq_enc'], row['id'])
            ),
            axis=1)

        train_valid_output_df = train_valid_data[
            ['nl_enc', 'nl_char_enc', 'ast_seq_enc', 'edge_order_seq', 'allowed_tokens',
             'ast_seq', 'snippet', 'id']]

        if self.eliminate_train_duplicates:
            train_valid_output_df = train_valid_output_df.drop_duplicates(subset=['nl_enc', 'ast_seq_enc'])

        save_json(train_valid_output_df, self.preproc_train_valid_data_output_file)
        stats_df = train_valid_data[['id', 'snippet', 'edge_counts', 'max_depth', 'ast_seq']]
        save_json(stats_df, self.preproc_train_valid_data_stats_output_file)

    def preproc_test_data(self, vocab_src, vocab_char):
        print()
        print(textwrap.dedent("""\
                    =======================
                    Preprocessing Test Data
                    =======================
                """))
        input_columns = ['id', 'nl', 'snippet', 'ast_seq', 'ast_seq_list']
        test_data = self.test_dataprovider.provide_data(columns=input_columns)

        test_data['nl_enc'] = self.encode_nl(test_data['nl'], vocab_src)
        test_data['nl_char_enc'] = self.encode_nl_char(test_data['nl'], vocab_src, vocab_char)
        output_columns = ['nl_enc', 'snippet', 'id', 'nl', 'nl_char_enc', 'ast_seq_list']
        dataframe = test_data[output_columns]
        save_json(dataframe, self.preproc_test_data_output_file)

    def is_valid_string(self, s):
        if not isinstance(s, (str, bytes)):
            return False
        try:
            s.encode('UTF-8').decode('UTF-8')
        except:
            print('Not UTF-8 encodable: %s' % ascii(s))
            return False

        try:
            s.encode('ISO-8859-1').decode('ISO-8859-1')
        except:
            print('Not ISO-8859-1 encodable: %s' % ascii(s))
            return False

        if isinstance(s, str) and '\0' in s:
            print('Contains null bytes: %s' % ascii(s))
            return False

        if isinstance(s, bytes) and b'\0' in s:
            print('Contains null bytes: %s' % ascii(s))
            return False

        return True

    def create_or_load_src_and_char_vocab(self):
        train_valid_data = self.train_valid_dataprovider.provide_data(columns=['nl', 'ast_seq_list_str'])
        src_vocab_exists = os.path.exists(self.vocabsrc_file)
        char_vocab_exists = os.path.exists(self.vocabchar_file)
        if not (src_vocab_exists and char_vocab_exists):
            train_valid_data_str_lit = pd.Series(s for strlist in train_valid_data['ast_seq_list_str'] for s in strlist)
            train_valid_data_src = itertools.chain(train_valid_data['nl'], train_valid_data_str_lit)
            train_valid_data_src = [s for s in train_valid_data_src if self.is_valid_string(s)]

        if not src_vocab_exists:
            assert not char_vocab_exists, "Character vocabulary exists but source vocabulary doesn't, might be inconsistent!"
            print('Creating source vocabulary', self.vocabsrc_file)
            # For creating a word vocab set add_dummy_prefix=True
            VocabularyCreator(train_valid_data_src,
                              vocab_file=self.vocabsrc_file,
                              vocab_size=self.vocabsrc_size,
                              vocab_type=self.vocabsrc_type,
                              max_sentencepiece_length=self.vocabsrc_max_sentencepiece_length,
                              split_by_whitespace=self.vocabsrc_split_by_whitespace,
                              split_digits=self.vocabsrc_split_digits,
                              treat_whitespace_as_suffix=self.vocabsrc_treat_whitespace_as_suffix,
                              byte_fallback=self.vocabsrc_byte_fallback,
                              add_dummy_prefix=self.vocabsrc_add_dummy_prefix,
                              user_defined_symbols=self.vocabsrc_user_defined_symbols
                              ).create_vocab()
        else:
            print('Source vocabulary already exists', self.vocabsrc_file)

        if not char_vocab_exists:
            assert not src_vocab_exists, "Source vocabulary exists but character vocabulary doesn't, might be inconsistent!"
            print('Creating character vocabulary', self.vocabchar_file)
            VocabularyCreator(train_valid_data_src,
                              vocab_file=self.vocabchar_file,
                              vocab_type='char',
                              add_dummy_prefix=False
                              ).create_vocab()
        else:
            print('Character vocabulary already exists', self.vocabchar_file)

        return self.load_vocab(self.vocabsrc_file), self.load_vocab(self.vocabchar_file)

    def create_or_load_tgt_vocab(self):
        if not os.path.exists(self.vocabtgt_file):
            assert not os.path.exists(
                self.grammar_graph_file), 'Grammar graph file already exists, but should not because the target vocabulary file does not exist'
            print('Creating target vocabulary', self.vocabsrc_file)

            train_valid_data = self.train_valid_dataprovider.provide_data(columns=['ast_seq_list_no_str'])
            train_valid_data_tgt = train_valid_data['ast_seq_list_no_str'].apply(lambda ast_seq: " ".join(ast_seq))

            VocabularyCreator(train_valid_data_tgt,
                              self.vocabtgt_file,
                              vocab_size=999999,
                              vocab_type='word').create_vocab()
        else:
            print('Target vocabulary already exists', self.vocabtgt_file)

        return self.load_vocab(self.vocabtgt_file)

    def load_vocab(self, vocab_file):
        vocab = VocabularyLoader(vocab_file=vocab_file).load_bpe_model()
        print(self.vocab_as_map(vocab))
        return vocab

    def vocab_as_map(self, vocab):
        return {id: vocab.IdToPiece(id) for id in range(0, vocab.vocab_size())}

    def create_grammar_graph(self, vocab_tgt, vocab_src):
        if not os.path.exists(self.grammar_graph_file):
            print('Creating grammar graph', self.grammar_graph_file)
            train_valid_data = self.train_valid_dataprovider.provide_data(columns=['id', 'nl', 'snippet', 'ast'])

            if self.language == 'python':
                grammar_creator = PythonGrammarGraphCreator(
                    vocab_tgt,
                    vocab_src,
                    all_strliteral_edges=self.all_strliteral_edges,
                    vocabsrc_add_dummy_prefix=self.vocabsrc_add_dummy_prefix,
                )
            elif self.language == 'lambdadcs':
                grammar_creator = AsdlGrammarGraphCreator(
                    vocab_tgt,
                    vocab_src,
                    self.lambdadcsgrammar,
                    all_strliteral_edges=self.all_strliteral_edges,
                    vocabsrc_add_dummy_prefix=self.vocabsrc_add_dummy_prefix,
                )

            created_grammar_graph = grammar_creator.create_graph(data=train_valid_data)
            grammar_creator.save_graph(self.grammar_graph_file)
            return created_grammar_graph
        else:
            print('Grammar graph already exists', self.grammar_graph_file)
            return None

    def encode_nl(self, nl, vocab_src):
        return nl.apply(lambda nl: tuple(vocab_src.encode(nl, add_bos=True, add_eos=True)))

    def encode_nl_char(self, nl, vocab_src, vocab_char):
        return nl.apply(lambda nl: self.preproc_char_seq(nl, vocab_src, vocab_char))

    def preproc_char_seq(self, nl, vocab_src, vocab_char):
        nl_dec = vocab_src.EncodeAsPieces(nl)
        nl_dec.append('')
        nl_dec.insert(0, '')
        return [tuple(vocab_char.encode(t, add_bos=False, add_eos=False)) for t in nl_dec]

    def encode_ast_seq(self, ast_seq, snippet, id, grammar_graph, vocab_src, vocab_tgt):
        try:
            sos_id = grammar_graph.graph['sentence_start_id']
            eos_id = grammar_graph.graph['sentence_end_id']
            ast_seq_enc = [sos_id]
            suffix = '#strliteral'
            vocabtgt_len = len(vocab_tgt)
            empty_strliteral_id = vocabtgt_len

            for node in ast_seq:
                if node.endswith(suffix):
                    str_literal = eval(node.split(suffix)[0])
                    if len(str_literal) == 0:
                        ast_seq_enc.append(empty_strliteral_id)
                    else:
                        vocab_ids = vocab_src.Encode(str_literal)
                        for vocab_id in vocab_ids:
                            if vocab_id == vocab_src.unk_id():
                                raise ValueError("Unknown string literal token %s in snippet %s with id %s" % (
                                    ascii(str_literal), ascii(snippet), id))
                            ast_seq_enc.append(vocab_id + vocabtgt_len)
                else:
                    vocab_ids = vocab_tgt.Encode(node)
                    if len(vocab_ids) != 1:
                        raise ValueError('More than 1 token in encoding of "%s":' % str(node), vocab_ids)
                    ast_seq_enc.append(vocab_ids[0])

            ast_seq_enc.append(eos_id)
            return tuple(ast_seq_enc)
        except:
            print("Error encoding AST sequence %s:" % ast_seq)
            traceback.print_exc(file=sys.stdout)
            return None

    def validate_python_ast_seq(self, codegenerator, ast_seq_enc, ast_obj, snippet, id):
        try:
            generated_code = codegenerator.generate_code(ast_seq_enc)
            astor_code = astor.to_source(ast_obj)
            if generated_code == astor_code:
                return ast_seq_enc
            elif generated_code is not None:
                if astor_code.replace('b"', '"').replace("b'", "'") != generated_code:
                    print("Generated code snippet doesn't match with original.",
                          " Id: %s, Snippet: %s, AST: %s" % (id, ascii(snippet), ast.dump(ast_obj)))
                    print("Expected:  " + ascii(astor_code))
                    print("Generated: " + ascii(generated_code))
                else:
                    print("Ignoring byte literal-only difference. Id: ", id)
                    return ast_seq_enc
        except:
            print("Generated AST seq is not valid. Id: %s, Snippet: %s, AST: %s" % (id, ascii(snippet),
                                                                                    ast.dump(ast_obj)))
            traceback.print_exc(file=sys.stdout)

        return None


def get_args():
    parser = argparse.ArgumentParser("Preproc data", fromfile_prefix_chars='@')
    parser.add_argument("--language", type=str, choices=['python', 'lambdadcs'], default='python')
    parser.add_argument("--lambdadcs-grammar-file", type=str)
    parser.add_argument("--train-valid-data-path", type=str, default="train-valid-data.json")
    parser.add_argument("--test-data-path", type=str, default="test-data.json")
    parser.add_argument("--preproc-train-valid-data-output-file", type=str, default="preproc-train-valid-data.jsonl")
    parser.add_argument("--preproc-test-data-output-file", type=str, default="preproc-test-data.jsonl")
    parser.add_argument("--preproc-train-valid-data-stats-output-file", type=str,
                        default="preproc-train-valid-stats.jsonl")
    parser.add_argument("--vocabsrc-size", type=int, default=800)
    parser.add_argument("--vocabsrc-type", type=str, choices=["bpe", "word", "unigram", "char"], default='bpe')
    parser.add_argument("--vocabsrc-file", type=str, default='vocabsrc.model')
    parser.add_argument("--vocabtgt-file", type=str, default='vocabtgt.model')
    parser.add_argument("--vocabchar-file", type=str, default='vocabchar.model')
    parser.add_argument("--grammar-graph-file", type=str, default='grammargraph.gpickle')
    parser.add_argument("--process-train-data", type=bool_str, default=False)
    parser.add_argument("--process-test-data", type=bool_str, default=False)
    parser.add_argument("--all-strliteral-edges", type=bool_str, default=False)
    parser.add_argument("--eliminate-train-duplicates", action='store_true')
    parser.add_argument("--validate-ast-seq-enc", type=bool_str, default=True)
    parser.add_argument("--reorder", choices=['none', 'atis'])
    parser.add_argument("--vocabsrc-max-sentencepiece-length", type=int, default=16)
    parser.add_argument("--vocabsrc-split-by-whitespace", type=bool_str, default=True)
    parser.add_argument("--vocabsrc-split-digits", type=bool_str, default=False)
    parser.add_argument("--vocabsrc-treat-whitespace-as-suffix", type=bool_str, default=False)
    parser.add_argument("--vocabsrc-byte-fallback", type=bool_str, default=False)
    parser.add_argument("--vocabsrc-add-dummy-prefix", type=bool_str, default=True)
    parser.add_argument("--vocabsrc-user-defined-symbols", type=str, default='')

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
    print("Preproc args:", vars(args))
    log.basicConfig(level=log.DEBUG)

    if args.language == 'python':
        lambdadcsgrammar = None
        train_valid_dataprovider = PreprocTrainPythonDataProvider(args.train_valid_data_path)
        test_dataprovider = PreprocTestPythonDataProvider(args.test_data_path)
    elif args.language == 'lambdadcs':
        lambdadcsgrammar = ASDLGrammar.from_text(open(args.lambdadcs_grammar_file).read())
        reorder_predicates = (args.reorder == 'atis')
        train_valid_dataprovider = PreprocTrainLambdaDCSDataProvider(args.train_valid_data_path,
                                                                     grammar=lambdadcsgrammar,
                                                                     reorder_predicates=reorder_predicates)
        test_dataprovider = PreprocTestLambdaDCSDataProvider(args.test_data_path, lambdadcsgrammar,
                                                             reorder_predicates=reorder_predicates)

    else:
        raise ValueError('Language not known: ' + args.language)

    preproc = Preproc(
        train_valid_dataprovider=train_valid_dataprovider,
        test_dataprovider=test_dataprovider,
        preproc_train_valid_data_output_file=args.preproc_train_valid_data_output_file,
        preproc_test_data_output_file=args.preproc_test_data_output_file,
        preproc_train_valid_data_stats_output_file=args.preproc_train_valid_data_stats_output_file,
        vocabsrc_file=args.vocabsrc_file,
        vocabtgt_file=args.vocabtgt_file,
        vocabsrc_size=args.vocabsrc_size,
        vocabsrc_type=args.vocabsrc_type,
        vocabchar_file=args.vocabchar_file,
        grammar_graph_file=args.grammar_graph_file,
        process_train_data=args.process_train_data,
        process_test_data=args.process_test_data,
        all_strliteral_edges=args.all_strliteral_edges,
        language=args.language,
        lambdadcsgrammar=lambdadcsgrammar,
        eliminate_train_duplicates=args.eliminate_train_duplicates,
        validate_ast_seq_enc=args.validate_ast_seq_enc,
        vocabsrc_max_sentencepiece_length=args.vocabsrc_max_sentencepiece_length,
        vocabsrc_split_by_whitespace=args.vocabsrc_split_by_whitespace,
        vocabsrc_split_digits=args.vocabsrc_split_digits,
        vocabsrc_treat_whitespace_as_suffix=args.vocabsrc_treat_whitespace_as_suffix,
        vocabsrc_byte_fallback=args.vocabsrc_byte_fallback,
        vocabsrc_add_dummy_prefix=args.vocabsrc_add_dummy_prefix,
        vocabsrc_user_defined_symbols=args.vocabsrc_user_defined_symbols
    )

    preproc.prepare_data()


if __name__ == '__main__':
    main()
