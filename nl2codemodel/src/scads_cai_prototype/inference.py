import argparse
from multiprocessing import freeze_support

import torch

from scads_cai_prototype.beamsearch import BeamSearch
from scads_cai_prototype.generator.codegenerator import PythonCodeGenerator
from scads_cai_prototype.grammar.grammargraphloader import GrammarGraphLoader
from scads_cai_prototype.nl2code import load_checkpoint
from scads_cai_prototype.preprocessing.preprocinf import Preprocinf


def get_args():
    parser = argparse.ArgumentParser("Infer", fromfile_prefix_chars='@')
    parser.add_argument("--test-model-path", type=str, default=None)
    parser.add_argument("--vocabsrc-file", type=str, default=None)
    parser.add_argument("--vocabchar-file", type=str, default=None)
    parser.add_argument("--grammar-graph-file", type=str, default='pythongrammar.json')
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-beam-length", type=int, default=30)
    parser.add_argument("--max-num-predicted-results", type=int, default=1)
    parser.add_argument("--beam-search-mode", choices=['full', 'reduced', 'scaled'], default='full')
    parser.add_argument("--disable-decoder-constraint-mask", type=bool_str, default=True)
    parser.add_argument("--keep-invalid-beamsearch-results", type=bool_str, default=False)

    parser.add_argument("--nl-input", type=str, default=None)

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

    freeze_support()

    system = load_checkpoint(args.test_model_path, {})

    model = system.model
    grammargraph = GrammarGraphLoader(args.grammar_graph_file).load_graph()
    codegenerator = PythonCodeGenerator(grammargraph)

    model.eval()

    beamsearch = BeamSearch(
        model=model,
        grammargraph=grammargraph,
        codegenerator=codegenerator,
        num_beams=args.num_beams,
        disable_constraint_mask=args.disable_decoder_constraint_mask,
        max_beam_length=args.max_beam_length,
        max_num_predicted_results=args.max_num_predicted_results,
        mode=args.beam_search_mode,
        keep_invalid_results=args.keep_invalid_beamsearch_results)

    preprocinf = Preprocinf(args.vocabsrc_file, args.vocabchar_file)
    nl_input = args.nl_input
    nl_seq = preprocinf.preproc(nl_input)
    nl_seq = torch.tensor(nl_seq)
    nl_seq = nl_seq.view(-1, 1)

    if not model.withcharemb:
        char_seqs = None
    else:
        char_seqs = preprocinf.preproc_char_seq(nl_input)
        char_seq_pad = preprocinf.pad_char_seq(char_seqs, model.max_charseq_len)
        char_seqs = char_seq_pad.view(-1, 1, model.max_charseq_len)

    with torch.no_grad():
        results = beamsearch.perform_beam_search(nl_seq, char_seqs, None)

    if results:
        print('Prediction(s):', [code_snippets for _, _, code_snippets in results])
    else:
        print('No results found')


if __name__ == '__main__':
    main()
