import argparse
import json

import torch

from nltocode.beamsearch import BeamSearch
from nltocode.generator.codegenerator import PythonCodeGenerator
from nltocode.grammar.grammargraphloader import GrammarGraphLoader
from nltocode.nl2code import load_checkpoint
from nltocode.preprocessing.preprocinf import Preprocinf

# import pickle5 as pickle

def get_args():
    parser = argparse.ArgumentParser("Attention Weights", fromfile_prefix_chars='@')

    # Input args
    parser.add_argument("--nl-input", type=str, default=None)

    # Attention weights args
    parser.add_argument("--attention-weights-mode", choices=['bygreedy', 'bymodel'], default='bygreedy')

    # Model args
    parser.add_argument("--test-model-path", type=str, default=None)
    parser.add_argument("--grammar-graph-file", type=str, default='pythongrammar.json')

    # Beam Search args
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-beam-length", type=int, default=30)
    parser.add_argument("--max-num-predicted-results", type=int, default=1)
    parser.add_argument("--beam-search-mode", choices=['full', 'reduced', 'scaled'], default='full')
    parser.add_argument("--disable-decoder-constraint-mask", type=bool_str, default=True)
    parser.add_argument("--keep-invalid-beamsearch-results", type=bool_str, default=False)

    # Preproc args
    parser.add_argument("--vocabsrc-file", type=str, default=None)
    parser.add_argument("--vocabchar-file", type=str, default=None)

    # Attention weights logging
    parser.add_argument("--self-attention-weights-file", type=str, default=None)
    parser.add_argument("--cross-attention-weights-file", type=str, default=None)
    parser.add_argument("--nl-ast-seq-file", type=str, default=None)

    return parser.parse_args()


def bool_str(val):
    if val == 'True':
        return True
    elif val == 'False':
        return False
    else:
        raise ValueError('Unexpected bool value: ', val)


def store_attention_weight(output, file):
    tgt, attention_weights = output
    attention_weights = torch.squeeze(attention_weights, 0)
    lists = attention_weights.tolist()
    write_to_file(lists, file, 'w')


def write_to_file(output, file, mode):
    json_str = json.dumps(output)

    with open(file, mode) as f:
        f.write(json_str)


def decode_nl_ast_seq(model, grammargraph, nl_seq, ast_seq):
    vocabtgt_size = model.vocabtgt_size
    vocabsrc_size = model.vocabsrc_size
    vocabtgt_size = vocabtgt_size - vocabsrc_size

    nl_seq_dec = [grammargraph.nodes[x + vocabtgt_size]['label'][:-len('#strliteral')] for x in nl_seq]

    ast_seq_dec = [grammargraph.nodes[x]['label'] for x in ast_seq]
    ast_seq_dec = [x[:-len('#strliteral')] if x.endswith('#strliteral') else x for x in ast_seq_dec]
    ast_seq_dec = [x[:-len('#literal')] if x.endswith('#literal') else x for x in ast_seq_dec]

    return {'nl_seq_dec': nl_seq_dec, 'ast_seq_dec': ast_seq_dec}


def attention_weights_by_greedy(args, system, nl_seq, char_seq_tensor):
    model = system.model
    grammargraph = GrammarGraphLoader(args.grammar_graph_file).load_graph()

    beamsearch = BeamSearch(
        model=model,
        grammargraph=grammargraph,
        codegenerator=PythonCodeGenerator(grammargraph),
        num_beams=1,
        disable_constraint_mask=args.disable_decoder_constraint_mask,
        max_beam_length=args.max_beam_length,
        max_num_predicted_results=1,
        mode=args.beam_search_mode,
        keep_invalid_results=args.keep_invalid_beamsearch_results)

    model.decoder.layers[1].self_attn.register_forward_hook(
        lambda self, input, output: store_attention_weight(output, args.self_attention_weights_file))
    model.decoder.layers[1].multihead_attn.register_forward_hook(
        lambda self, input, output: store_attention_weight(output, args.cross_attention_weights_file))

    nl_seq_tensor = torch.tensor(nl_seq).view(-1, 1)

    with torch.no_grad():
        results = beamsearch.perform_beam_search(nl_seq_tensor, char_seq_tensor)

    ast_seq, _, _ = results[0]

    write_to_file(decode_nl_ast_seq(model, grammargraph, nl_seq, ast_seq), args.nl_ast_seq_file, 'w')


def beam_search(args, system, nl_seq):
    model = system.model
    grammargraph = GrammarGraphLoader(args.grammar_graph_file).load_graph()

    # to open pickle created with Python 3.8
    #with open(args.grammar_graph_file, "rb") as fh:
        #grammargraph = pickle.load(fh)

    beamsearch = BeamSearch(
        model=model,
        grammargraph=grammargraph,
        codegenerator=PythonCodeGenerator(grammargraph),
        num_beams=args.num_beams,
        disable_constraint_mask=args.disable_decoder_constraint_mask,
        max_beam_length=args.max_beam_length,
        max_num_predicted_results=args.max_num_predicted_results,
        mode=args.beam_search_mode,
        keep_invalid_results=args.keep_invalid_beamsearch_results)

    nl_seq_tensor = torch.tensor(nl_seq).view(-1, 1)

    with torch.no_grad():
        results = beamsearch.perform_beam_search(nl_seq_tensor, None, return_edge_order_seqs=True)

    ast_seq, _, _, edge_order_seq = results[0]

    write_to_file(decode_nl_ast_seq(model, grammargraph, nl_seq, ast_seq), args.nl_ast_seq_file, 'w')

    ast_seq_tensor = torch.tensor(ast_seq, dtype=torch.long)
    ast_seq_tensor = ast_seq_tensor.view(-1, 1)

    edge_order_seq_tensor = beamsearch.edge_path_transform(
        [eo_path for eo_path, node_type in edge_order_seq if
         node_type in ('node', 'strliteral', 'literal', 'special')])
    edge_order_seq_tensor = edge_order_seq_tensor.unsqueeze(1)

    return ast_seq_tensor, edge_order_seq_tensor


def attention_weights_by_model(args, system, nl_seq):
    model = system.model
    ast_seq_tensor, edge_order_seq_tensor = beam_search(args, system, nl_seq)
    nl_seq_tensor = torch.tensor(nl_seq).view(-1, 1)

    model.decoder.layers[1].self_attn.register_forward_hook(
        lambda self, input, output: store_attention_weight(output, args.self_attention_weights_file))
    model.decoder.layers[1].multihead_attn.register_forward_hook(
        lambda self, input, output: store_attention_weight(output, args.cross_attention_weights_file))

    model(nl_seq_tensor, None, ast_seq_tensor[:-1], edge_order_seq_tensor[:-1])


def main():
    args = get_args()

    system = load_checkpoint(args.test_model_path, test_args={})
    system.eval()

    preprocinf = Preprocinf(args.vocabsrc_file, args.vocabchar_file)

    nl_input = args.nl_input
    nl_seq = preprocinf.preproc(nl_input)
    if system.model.withcharemb:
        char_seq = preprocinf.preproc_char_seq(nl_input)
        char_seq_padded = preprocinf.pad_char_seq(char_seq, system.model.max_charseq_len)
        char_seq_padded = char_seq_padded.view(-1, 1, system.model.max_charseq_len)
    else:
        char_seq_padded = None

    if args.attention_weights_mode == 'bygreedy':
        attention_weights_by_greedy(args, system, nl_seq, char_seq_padded)
    elif args.attention_weights_mode == 'bymodel':
        attention_weights_by_model(args, system, nl_seq)
    else:
        print(args.attention_weights_mode, 'mode not supported.')


if __name__ == '__main__':
    main()
