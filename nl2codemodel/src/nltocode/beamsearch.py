import logging as log
import math
from itertools import islice

import torch

from nltocode.dataset import EdgePathTransform
from nltocode.grammar.grammargraphvisitor import GrammarGraphVisitor


class BeamSearch():
    def __init__(self,
                 model,
                 grammargraph,
                 codegenerator,
                 num_beams,
                 disable_constraint_mask,
                 max_beam_length,
                 max_num_predicted_results,
                 batch_size=None,
                 mode='full',
                 treat_empty_results_as_invalid=False,
                 keep_invalid_results=False,
                 report_every_n_tokens=None,
                 report_k_best_beams=None,
                 ):
        '''
        For default beam search behaviour, set beam_search_mode = 'reduced' and max_num_predicted_results == num_beams

        batch_size defaults to num_beams and needs to be reduced if out-of-memory errors occur
        '''
        super().__init__()
        self.model = model
        self.grammargraph = grammargraph

        self.sos_id = self.grammargraph.graph['sentence_start_id']
        self.eos_id = self.grammargraph.graph['sentence_end_id']
        self.list_end_id = self.grammargraph.graph['list_end_id']

        self.grammargraphvisitor = GrammarGraphVisitor(self.grammargraph)
        self.codegenerator = codegenerator

        self.k_beams = num_beams
        self.disable_constraint_mask = disable_constraint_mask
        self.max_beam_length = max_beam_length
        self.edge_path_transform = EdgePathTransform(model.max_path_depth)

        self.max_num_predicted_results = max_num_predicted_results
        self.batch_size = batch_size if batch_size is not None else num_beams
        self.mode = mode
        self.treat_empty_results_as_invalid = treat_empty_results_as_invalid
        self.keep_invalid_results = keep_invalid_results
        self.report_every_n_tokens = report_every_n_tokens
        self.report_k_best_beams = report_k_best_beams

    def perform_beam_search(self, nl_seq, char_seq=None, return_edge_order_seqs=False):
        device = nl_seq.device

        nl_seq_transposed = nl_seq.transpose(0, 1).tolist()
        nl_seq_str = [self.decode_nl_seq(n) for n in nl_seq_transposed]

        log.info("Input device = %s, shape = %s", device, nl_seq.size())
        log.info("Input sequence(s): %s" % nl_seq_transposed)
        log.info("Input text(s): %s" % nl_seq_str)

        self.edge_path_transform.device = device

        nl_len = nl_seq.size(0)
        nl_count = nl_seq.size(1)
        assert nl_count == 1, "Only one NL input can be processed at a time"
        if char_seq is not None:
            assert char_seq.size(0) == nl_len
            assert char_seq.size(1) == nl_count
            max_charseq_len = char_seq.size(2)
        else:
            max_charseq_len = None

        src_encoding = self.model.encode_src(nl_seq, char_seq)
        d_model = src_encoding.size(2)

        ast_seq_tensor = torch.tensor([[self.sos_id]], dtype=torch.long, device=device)
        scores = torch.tensor([0.0], device=device)
        paths = [[]]
        edge_order_seqs = [[]]
        edge_order_paths = [[]]

        result_beams = []
        worst_score_to_be_returned = float("inf")

        ast_seq_len = ast_seq_tensor.size(0)
        num_live_beams = ast_seq_tensor.size(1)
        num_active_beams = self.compute_num_active_beams(result_beams)

        while ast_seq_len <= self.max_beam_length and num_live_beams > 0 and num_active_beams > 0:
            nl_seq_expanded = nl_seq.expand((nl_len, num_live_beams))
            char_seq_expanded = expand_or_none(char_seq, (nl_len, num_live_beams, max_charseq_len))
            src_encoding_expanded = src_encoding.expand((nl_len, num_live_beams, d_model))

            allowed_next_token_lists, edge_order_paths, edge_order_seqs, paths = self.apply_visit_grammargraph(
                ast_seq_tensor, edge_order_paths, edge_order_seqs, num_live_beams, paths)
            edge_order_seqs_transformed = [
                self.edge_path_transform(filter_ast_nodes(seq)).to(device)
                for seq in edge_order_seqs
            ]
            edge_order_seq_tensor = torch.stack(edge_order_seqs_transformed, dim=1)
            log_probs = self.model(nl_seq_expanded, char_seq_expanded, ast_seq_tensor, edge_order_seq_tensor,
                                   src_encoding_expanded)

            # Dim: batch_size x vocab_size
            if not self.disable_constraint_mask:
                next_tokens_log_probs = compute_next_token_allowed_log_probs(log_probs, allowed_next_token_lists)
            else:
                next_tokens_log_probs = log_probs[-1, :, :]

            top_k_values, top_k_vocab_ids = torch.topk(next_tokens_log_probs, num_active_beams, dim=-1, sorted=False)
            top_k_shape = top_k_values.shape

            top_k_prev_scores = scores.unsqueeze(-1).expand(top_k_shape)
            top_k_scores = top_k_prev_scores - top_k_values

            top_k_is_retained = (top_k_scores <= worst_score_to_be_returned).logical_and(top_k_scores != float('inf'))

            retained_beam_ids = torch.arange(num_live_beams).unsqueeze(-1).expand(top_k_shape)[top_k_is_retained]
            retained_vocab_ids: torch.Tensor = top_k_vocab_ids[top_k_is_retained]
            retained_scores = top_k_scores[top_k_is_retained]
            retained_is_finished = retained_vocab_ids == self.eos_id
            retained_is_unfinished = retained_is_finished.logical_not()

            finished_beam_ids = retained_beam_ids[retained_is_finished]
            for finished_beam_id in finished_beam_ids.tolist():
                finished_ast_seq_tensor_prefix = ast_seq_tensor[:, finished_beam_id]
                finished_vocab_id = torch.tensor([self.eos_id], dtype=torch.long, device=device)
                finished_ast_seq = torch.cat((finished_ast_seq_tensor_prefix, finished_vocab_id), dim=0)
                finished_score = scores[finished_beam_id]
                finished_ast_seq_list = finished_ast_seq.tolist()
                code_snippet = self.codegenerator.generate_code(finished_ast_seq_list)

                if self.treat_empty_results_as_invalid:
                    code_snippet_is_valid = bool(code_snippet)
                else:
                    code_snippet_is_valid = code_snippet is not None

                if code_snippet_is_valid or self.keep_invalid_results:
                    result_beam = finished_ast_seq_list, finished_score, code_snippet
                    if return_edge_order_seqs:
                        finished_path = paths[finished_beam_id]
                        finished_edge_order_seq = edge_order_seqs[finished_beam_id]
                        finished_edge_order_path = edge_order_paths[finished_beam_id]
                        _, _, edge_order_seq, _ = self.visit_grammargraph(
                            torch.tensor(self.eos_id, device=device),
                            finished_path,
                            finished_edge_order_seq,
                            finished_edge_order_path
                        )
                        result_beam = result_beam + (edge_order_seq,)

                    log.debug('Finished beam: %s', result_beam)
                    result_beams.append(result_beam)
                    result_beams.sort(key=lambda tup: tup[1])
                    # result_score_stats = pandas.Series([tup[1] for tup in result_beams], dtype=float)
                    # log.debug("Result score stats: %s", result_score_stats.describe())

                    if len(result_beams) >= self.max_num_predicted_results:
                        result_beams = result_beams[:self.max_num_predicted_results]
                        worst_beam_to_be_returned = result_beams[-1]
                        worst_score_to_be_returned = worst_beam_to_be_returned[1]
                        log.debug("New maximum score to be returned: %.3f", worst_score_to_be_returned)
                else:
                    log.debug("Illegal AST seq: %s", self.decode_ast_seq(finished_ast_seq_list))

            unfinished_scores = retained_scores[retained_is_unfinished]
            unfinished_beam_ids = retained_beam_ids[retained_is_unfinished]
            unfinished_vocab_ids = retained_vocab_ids[retained_is_unfinished]

            num_live_beams = min(num_active_beams, unfinished_scores.size(0))

            live_scores, unfinished_live_indexes = unfinished_scores.topk(num_live_beams, largest=False, sorted=True)

            live_beam_ids = unfinished_beam_ids[unfinished_live_indexes]

            live_ast_seq_tensor = ast_seq_tensor[:, live_beam_ids]

            live_vocab_ids = unfinished_vocab_ids[unfinished_live_indexes]

            if self.report_every_n_tokens and (ast_seq_len % self.report_every_n_tokens == 0):
                self.log_report(ast_seq_tensor, log_probs, scores)

            live_beam_ids_list = live_beam_ids.tolist()

            ast_seq_tensor = torch.cat((live_ast_seq_tensor, live_vocab_ids.unsqueeze(0)), dim=0)
            paths = [paths[i] for i in live_beam_ids_list]
            edge_order_seqs = [edge_order_seqs[i] for i in live_beam_ids_list]
            edge_order_paths = [edge_order_paths[i] for i in live_beam_ids_list]

            scores = live_scores
            ast_seq_len = ast_seq_tensor.size(0)
            num_live_beams = ast_seq_tensor.size(1)
            num_active_beams = self.compute_num_active_beams(result_beams)

        log.debug('BEAMS: %s', result_beams)
        return result_beams

    def log_report(self, ast_seq_tensor, log_probs, scores):
        ast_seq_len = ast_seq_tensor.size(0)
        num_live_beams = ast_seq_tensor.size(1)
        log.info("Current AST sequence length: %s" % ast_seq_len)
        num_reported_beams = self.report_k_best_beams or num_live_beams
        reported_ast_seqs = ast_seq_tensor.transpose(0, 1).tolist()
        reported_scores = scores.tolist()
        for i, reported_ast_seq in islice(enumerate(reported_ast_seqs), num_reported_beams):
            token_scores = score_ast_seq_tensor(ast_seq_tensor, log_probs)
            reported_ast_seq_dec = self.decode_ast_seq(reported_ast_seq)
            reported_token_scores = [0] + token_scores[:, i].tolist()
            reported_token_score_strs = ["%4.1e" % token_score for token_score in reported_token_scores]
            reported_beam_tokens_with_scores = list(zip(reported_ast_seq_dec, reported_token_score_strs))
            log.info("Score %9.6f: %s" % (reported_scores[i], reported_beam_tokens_with_scores))

    def decode_nl_seq(self, nl_seq):
        return decode_nl_seq(nl_seq, self.grammargraph, self.model.vocabtgt_size - self.model.vocabsrc_size)

    def decode_ast_seq(self, ast_seq):
        return decode_ast_seq(ast_seq, self.grammargraph)

    def compute_num_active_beams(self, result_beams):
        if self.mode == 'full':
            return self.k_beams
        elif self.mode == 'reduced':
            return max(0, self.k_beams - len(result_beams))
        elif self.mode == 'scaled':
            return math.ceil(self.k_beams * (1 - len(result_beams) / self.max_num_predicted_results))
        else:
            raise ValueError("Unknown beam search mode '%s'" % self.mode)

    def apply_visit_grammargraph(self, ast_seq_tensor, edge_order_paths, edge_order_seqs, num_live_beams, paths):
        graphvisitor_results = (
            self.visit_grammargraph(
                ast_seq_tensor[-1, i],
                paths[i],
                edge_order_seqs[i],
                edge_order_paths[i]
            )
            for i in range(num_live_beams)
        )
        allowed_next_token_lists, paths, edge_order_seqs, edge_order_paths = zip(*graphvisitor_results)
        return allowed_next_token_lists, edge_order_paths, edge_order_seqs, paths

    def visit_grammargraph(self, node, path, edge_order_seq, edge_order_path):
        return self.grammargraphvisitor.visit_graph_edge_order_path_beam_search(
            node.item(),
            path.copy(),
            edge_order_seq.copy(),
            edge_order_path.copy()
        )


def compute_next_token_allowed_log_probs(log_probs, allowed_next_token_lists):
    allowed_log_probs = torch.full(log_probs.shape[1:], fill_value=float("-inf"), device=log_probs.device)
    for i, allowed_next_token_list in enumerate(allowed_next_token_lists):
        allowed_next_tokens = torch.tensor(allowed_next_token_list, dtype=torch.long,
                                           device=log_probs.device)
        allowed_log_probs[i, allowed_next_tokens] = log_probs[-1, i, allowed_next_tokens]
        if allowed_next_tokens.size(0) == 0:
            log.warning("No allowed successor tokens in beam %s at step %s", i, log_probs.size(0))
    return allowed_log_probs


def expand_or_none(t: torch.Tensor, shape):
    if t is not None:
        char_seq_expanded = t.expand(shape)
    else:
        char_seq_expanded = None
    return char_seq_expanded


def decode_nl_seq(nl_seq, grammargraph, vocabsrc_offset):
    suffix_length = 11
    nl_seq_dec = [grammargraph.nodes[vocabsrc_offset + t]['label'][:-suffix_length] for t in nl_seq]
    nl_seq_str = ''.join(nl_seq_dec[1:-1]).replace('▁', ' ')
    return nl_seq_str


def decode_ast_seq(ast_seq, grammargraph):
    return [grammargraph.nodes[t]['label'] for t in ast_seq]


def filter_ast_nodes(edge_order_seq):
    return [
        eo_path
        for eo_path, node_type
        in edge_order_seq
        if node_type in ('node', 'strliteral', 'literal', 'special')
    ]


def score_ast_seq_tensor(ast_seq_tensor, log_probs):
    return -log_probs.gather(-1, ast_seq_tensor[1:, :].unsqueeze(-1)).squeeze(-1)
