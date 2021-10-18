import argparse
import logging as log
import re
from collections import defaultdict
from difflib import SequenceMatcher
from os.path import commonprefix

import pandas as pd
import torch
from torch.nn.functional import nll_loss, log_softmax

from nltocode.datamodule import NL2CodeTrainDataModule
from nltocode.grammar.grammargraphloader import GrammarGraphLoader
from nltocode.grammar.grammargraphvisitor import AbstractGrammarGraphVisitor
from nltocode.nl2code import load_checkpoint
from nltocode.preprocessing.filehandling import save_json


def score(model_path, train_data_args, inference_args, output_filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    if device.type == 'cuda':
        current_device = torch.cuda.current_device()
        print('Current device: ', current_device)
        print('Current device name: ', torch.cuda.get_device_name(current_device))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    system = load_checkpoint(model_path, test_args=inference_args).to(device)
    model = system.model

    data_module = NL2CodeTrainDataModule(
        **train_data_args,
        max_path_depth=model.max_path_depth,
        path_multiple=model.path_multiple,
        max_charseq_len=model.max_charseq_len if model.withcharemb else None,
    )

    data_module.setup('fit')
    system.setup('test')
    system.eval()

    dataloader = data_module.val_dataloader()

    grammar_graph_file = inference_args.get('grammar_graph_file')
    grammar_graph = GrammarGraphLoader(grammar_graph_file).load_graph()
    grammar_graph_visitor = AbstractGrammarGraphVisitor(grammar_graph)

    outputs = []
    with torch.no_grad():
        i = 0
        for batch in dataloader:
            nl_seqs, _, ast_seqs, _ = batch

            beamsearch_predictions = system.beamsearch.perform_beam_search(nl_seqs.to(device))

            #prob_datapoint = comp_probs(system.model, nl_seqs, char_seqs, ast_seqs, edge_order_seqs)
            prefix_datapoint = comp_longest_prefix(system, beamsearch_predictions, ast_seqs)
            diff_analysis_res = perf_diff_analysis(beamsearch_predictions, grammar_graph_visitor, ast_seqs)

            #datapoint = {**prob_datapoint, **prefix_datapoint, **diff_analysis_res}
            datapoint = { **prefix_datapoint, **diff_analysis_res}
            outputs.append(datapoint)
            i += 1

        out_df = pd.DataFrame(outputs)
        save_json(out_df, output_filename)

def comp_longest_prefix(system, beamsearch_predictions, ast_seqs):
    expected_ast_seq = ast_seqs.view(-1).tolist()
    ltgt = system.model.vocabtgt_size - system.model.vocabsrc_size
    replaced_filtered_expected_ast_seq = filter_ast_seq(expected_ast_seq, ltgt)

    if len(beamsearch_predictions) == 0:
        beamsearch_ast_seq, score, code_snippet = [], float('inf'), ''
    else:
        beamsearch_ast_seq, score, code_snippet = beamsearch_predictions[0]

    replaced_filtered_beamsearch_ast_seq = filter_ast_seq(beamsearch_ast_seq, ltgt)
    common_prefix = commonprefix((expected_ast_seq, beamsearch_ast_seq))
    replaced_common_prefix = commonprefix((replaced_filtered_expected_ast_seq, replaced_filtered_beamsearch_ast_seq))

    return {
        'replaced_common_prefix_len': len(replaced_common_prefix),
        'replaced_filtered_expected_ast_seq_len': len(replaced_filtered_expected_ast_seq),
        'replaced_filtered_beamsearch_ast_seq_len': len(replaced_filtered_beamsearch_ast_seq),

        'common_prefix_len': len(common_prefix),
        'expected_ast_seq_len': len(expected_ast_seq),
        'beamsearch_ast_seq_len': len(beamsearch_ast_seq),

        'expected_ast_seq': expected_ast_seq,
        'beamsearch_ast_seq': beamsearch_ast_seq,
        'replaced_filtered_expected_ast_seq': replaced_filtered_expected_ast_seq,
        'replaced_filtered_beamsearch_ast_seq': replaced_filtered_beamsearch_ast_seq,
    }


def perf_diff_analysis(beamsearch_predictions, grammar_graph_visitor, ast_seqs):
    ast_seq_exp_enc = ast_seqs.view(-1).tolist()

    if len(beamsearch_predictions) == 0:
        ast_seq_pred_enc, score, code_snippet = [], float('inf'), ''
    else:
        ast_seq_pred_enc, score, code_snippet = beamsearch_predictions[0]

    ast_seq_exp_dec, ast_seq_exp_type = decode_ast_seq_as_char(ast_seq_exp_enc, grammar_graph_visitor)
    ast_seq_pred_dec, ast_seq_pred_type = decode_ast_seq_as_char(ast_seq_pred_enc, grammar_graph_visitor)

    s = SequenceMatcher(None, ast_seq_exp_dec, ast_seq_pred_dec)
    diff_analysis = defaultdict(list)

    for tag, i1, i2, j1, j2 in s.get_opcodes():
        diff_analysis[str(tag)].append(
            {'pos_exp_i1': i1,
             'pos_exp_i2': i2,
             'pos_pred_j1': j1,
             'pos_pred_j2': j2,
             'val_exp': ast_seq_exp_dec[i1:i2],
             'val_pred': ast_seq_pred_dec[j1:j2],
             'type_exp': ast_seq_exp_type[i1:i2],
             'type_pred': ast_seq_pred_type[j1:j2]
             })

    return {
        'diff_analysis_res': diff_analysis,
        'ast_seq_exp_dec': ast_seq_exp_dec,
        'ast_seq_exp_type': ast_seq_exp_type,
        'ast_seq_pred_dec': ast_seq_pred_dec,
        'ast_seq_pred_type': ast_seq_pred_type
    }

def comp_probs(model, nl_seqs, char_seqs, ast_seqs, edge_order_seqs):
    predicted_log_probs = predict(model, nl_seqs, char_seqs, ast_seqs, edge_order_seqs)
    expected_ast_seq = ast_seqs[1:, :].view(-1)
    expected_scores = compute_score(predicted_log_probs, expected_ast_seq, model.vocab_pad_id)
    predicted_ast_seq = torch.argmax(predicted_log_probs, dim=-1)
    predicted_scores = torch.max(predicted_log_probs, -1).values

    return {
        'target_expected_scores_sum': expected_scores.sum().item(),
        'target_predicted_scores_sum': predicted_scores.sum().item(),

        'nl_length': nl_seqs.size(0),
        'target_ast_seq_length': ast_seqs.size(0),
        'target_expected_scores': expected_scores.tolist(),
        'target_predicted_scores': predicted_scores.tolist(),
        'target_predicted_ast_seq': predicted_ast_seq.tolist(),
    }


def decode_ast_seq_as_char(ast_seq_enc, grammar_graph_visitor):
    ast_seq_dec = []
    ast_seq_type = []

    for token in ast_seq_enc:
        token_dec = grammar_graph_visitor.get_node_label(token)
        token_dec_ = re.sub(r'|'.join(map(re.escape, ['#strliteral', '#literal'])), '', token_dec)
        token_type = grammar_graph_visitor.get_node_type(token)

        ast_seq_dec.append(token_dec_)
        ast_seq_type.append(token_type)

    return ast_seq_dec, ast_seq_type


def decode_ast_seq(ast_seq_enc, grammar_graph_visitor):
    ast_seq_dec = []
    ast_seq_type = []
    i = 0

    while i < len(ast_seq_enc):
        token = ast_seq_enc[i]
        token_dec = grammar_graph_visitor.get_node_label(token)
        token_dec_ = re.sub(r'|'.join(map(re.escape, ['#strliteral', '#literal'])), '', token_dec)
        token_type = grammar_graph_visitor.get_node_type(token)
        concat_strlit_token = []

        while token_type == 'strliteral':
            concat_strlit_token.append(token_dec_)
            i = i + 1
            token = ast_seq_enc[i]
            token_dec = grammar_graph_visitor.get_node_label(token)
            token_dec_ = re.sub(r'|'.join(map(re.escape, ['#strliteral', '#literal'])), '', token_dec)
            token_type = grammar_graph_visitor.get_node_type(token)

        if concat_strlit_token:
            ast_seq_dec.append(''.join(concat_strlit_token))
            ast_seq_type.append('strliteral')

        ast_seq_dec.append(token_dec_)
        ast_seq_type.append(token_type)
        i = i + 1

    return ast_seq_dec, ast_seq_type


def filter_ast_seq(ast_seq_list, tgt_vocab_size):
    replaced_ast_seq = [tgt_vocab_size + 1 if vocab_id > tgt_vocab_size else vocab_id for vocab_id in ast_seq_list]
    replaced_filtered_ast_seq = []
    for i in replaced_ast_seq:
        if (i != tgt_vocab_size + 1) or (len(replaced_filtered_ast_seq) == 0) or (
                replaced_filtered_ast_seq[-1] != tgt_vocab_size + 1):
            replaced_filtered_ast_seq.append(i)

    return replaced_filtered_ast_seq


def compute_score(log_probs, sequence, pad_id):
    # nll_loss computes the score of each element in `sequence` according to the `log_probs`
    return -nll_loss(log_probs, sequence, ignore_index=pad_id, reduction='none').view(-1)


def predict(model, nl_seqs, char_seqs, ast_seqs, paths_seqs):
    print("shapes: ast_seq %s, paths %s" % (ast_seqs.shape, paths_seqs.shape))
    print("ast_seq %s" % ast_seqs.view(-1).tolist())
    print("paths %s" % paths_seqs.squeeze(1).tolist())
    logits = model(nl_seqs, char_seqs, ast_seqs[:-1, :], paths_seqs[:-1, :])
    output_dim = logits.shape[-1]
    logits = logits.view(-1, output_dim)
    log_probs = log_softmax(logits, 1)
    return log_probs


def get_args():
    parser = argparse.ArgumentParser("Estimate sequence scores", fromfile_prefix_chars='@')

    # Dataset parameters
    parser.add_argument("--preproc-data-path", type=str)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--grammar-graph-file", type=str)
    parser.add_argument("--output-file-path", type=str)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--max-beam-length", type=int, default=1000)
    parser.add_argument("--disable-decoder-constraint-mask", action='store_true')

    return parser.parse_args()


def main():
    torch.set_printoptions(precision=3, threshold=10240, linewidth=100000)
    args = get_args()
    print("Score args:", vars(args))
    log.basicConfig(level='DEBUG')

    inference_args = {
        'grammar_graph_file': args.grammar_graph_file,
        'target_language': 'python',
        'num_beams': args.num_beams,
        'disable_decoder_constraint_mask': args.disable_decoder_constraint_mask,
        'max_beam_length': args.max_beam_length,
        'max_num_predicted_results': 1,
        'beam_search_mode': 'full',
        'keep_invalid_beamsearch_results': False,
        'validate_parsability': False
    }
    train_data_args = {
        'batch_size': 1,
        'num_dataloader_workers': 1,
        'train_valid_data_path': args.preproc_data_path,
        'train_split': 0.0,
        'val_split': 1.0,
        'max_src_sentence_length': 10000,
        'max_tgt_sentence_length': 10000,
    }
    score(args.model_path, train_data_args, inference_args, args.output_file_path)


if __name__ == '__main__':
    main()
