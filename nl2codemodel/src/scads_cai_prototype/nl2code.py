import logging as log
from typing import Dict

import pandas as pd
import torch
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.saving import ModelIO
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, LambdaLR

from scads_cai_prototype.beamsearch import BeamSearch
from scads_cai_prototype.generator.astseqgenerator import AstSeqCodeGenerator
from scads_cai_prototype.generator.codegenerator import PythonCodeGenerator
from scads_cai_prototype.grammar.grammargraphloader import GrammarGraphLoader
from scads_cai_prototype.transformer import Transformer


class NL2CodeSystem(LightningModule):
    def __init__(self,
                 model_hparams,
                 learning_rate,
                 label_smoothing=None,
                 scheduler=None,
                 step_size=30,
                 gamma=0.1,
                 warmup_steps=4000,
                 sc_last_epoch=-1,
                 adam_beta_1=0.9,
                 adam_beta_2=0.999,
                 adam_epsilon=1e-8,
                 # Test-only parameters
                 grammar_graph_file=None,
                 target_language='python',
                 num_beams=None,
                 disable_decoder_constraint_mask=False,
                 max_beam_length=None,
                 max_num_predicted_results=None,
                 beam_search_mode=None,
                 treat_empty_beamsearch_results_as_invalid=False,
                 keep_invalid_beamsearch_results=False,
                 validate_parsability=True,
                 report_beams_every_n_tokens=None,
                 report_k_best_beams=None,
                 target_output_file=None,
                 train_data_args=None,
                 ):
        super().__init__()
        super_fields = list(self.__dict__.keys())

        self.model_hparams = model_hparams

        self.learning_rate = learning_rate
        self.label_smoothing = label_smoothing

        self.scheduler = scheduler
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_steps = warmup_steps
        self.sc_last_epoch = sc_last_epoch

        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.adam_epsilon = adam_epsilon

        self.train_data_args = train_data_args

        self.grammar_graph_file = grammar_graph_file
        self.target_language = target_language
        self.num_beams = num_beams
        self.disable_decoder_constraint_mask = disable_decoder_constraint_mask
        self.max_beam_length = max_beam_length
        self.max_num_predicted_results = max_num_predicted_results
        self.beam_search_mode = beam_search_mode
        self.treat_empty_beamsearch_results_as_invalid = treat_empty_beamsearch_results_as_invalid
        self.keep_invalid_beamsearch_results = keep_invalid_beamsearch_results
        self.validate_parsability = validate_parsability
        self.report_beams_every_n_tokens = report_beams_every_n_tokens
        self.report_k_best_beams = report_k_best_beams
        self.target_output_file = target_output_file

        hparam_names = [name for name in self.__dict__.keys() if name not in super_fields]
        self.save_hyperparameters(*hparam_names)

        self.model = Transformer(**model_hparams)

        # (src, char, tgt, edge_path_seqs, logits_token_mask)
        src_example = torch.zeros(dtype=torch.long, size=(2, 3))
        char_example = torch.zeros(dtype=torch.long,
                                   size=(2, 3, self.model.max_charseq_len)) if self.model.withcharemb else None
        tgt_example = torch.zeros(dtype=torch.long, size=(4, 3))
        edge_path_seq_example = torch.zeros(dtype=torch.long, size=(4, 3, self.model.max_path_depth))
        self.example_input_array = (src_example, char_example, tgt_example, edge_path_seq_example)

        self.beamsearch = None
        self.ast_seq_generator = None

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def setup(self, stage=None):
        if stage == 'test':
            grammargraph = GrammarGraphLoader(self.grammar_graph_file).load_graph()
            pad_id = grammargraph.graph['pad_id']
            if pad_id is not None and pad_id != self.model.vocab_pad_id:
                raise ValueError('Padding id saved in graph differs from padding id saved in model')

            codegenerator = self.create_code_generator(grammargraph)

            # Default max_beam_length to max_tgt_sentence_length used in training plus 10%
            self.beamsearch = BeamSearch(
                model=self.model,
                grammargraph=grammargraph,
                codegenerator=codegenerator,
                num_beams=self.num_beams,
                disable_constraint_mask=self.disable_decoder_constraint_mask,
                max_beam_length=self.max_beam_length,
                max_num_predicted_results=self.max_num_predicted_results,
                mode=self.beam_search_mode,
                treat_empty_results_as_invalid=self.treat_empty_beamsearch_results_as_invalid,
                keep_invalid_results=self.keep_invalid_beamsearch_results,
                report_every_n_tokens=self.report_beams_every_n_tokens,
                report_k_best_beams=self.report_k_best_beams,
            )
            self.ast_seq_generator = AstSeqCodeGenerator(grammargraph)

    def create_code_generator(self, grammargraph):
        if self.target_language == 'python':
            validate_parsability = self.validate_parsability and not self.keep_invalid_beamsearch_results
            return PythonCodeGenerator(grammargraph, validate_parsability=validate_parsability)
        elif self.target_language == 'astseq':
            return AstSeqCodeGenerator(grammargraph)
        else:
            raise ValueError('Unsupported target language %s' % self.target_language)

    def training_step(self, batch, batch_idx):
        train_loss = self.compute_train_val_loss(batch)
        train_perplexity = torch.exp(train_loss)

        return {'loss': train_loss, 'train_perplexity': train_perplexity}

    def compute_train_val_loss(self, batch, loss_reduction='mean'):
        if not self.label_smoothing:
            return self.compute_train_val_loss_without_label_smoothing(batch, loss_reduction)
        else:
            return self.compute_train_val_loss_with_label_smoothing(batch, loss_reduction)

    def compute_train_val_loss_without_label_smoothing(self, batch, loss_reduction):
        nl_seqs, nl_char_seqs, ast_seqs, edge_paths_seqs = batch
        log_probs = self(nl_seqs, nl_char_seqs, ast_seqs[:-1, :], edge_paths_seqs[:-1, :, :])
        output_dim = log_probs.shape[-1]
        log_probs = log_probs.view(-1, output_dim)
        ast_seqs = ast_seqs[1:, :].view(-1)
        loss = F.nll_loss(log_probs, ast_seqs, ignore_index=self.model.vocab_pad_id, reduction=loss_reduction)
        # Could re-weight losses here like this if needed:
        # losses = nll_loss(log_probs, ast_seqs, ignore_index=self.model.vocab_pad_id, reduction='none').view(logits.shape[:-1])
        # Interesting for this: torch.cumsum
        return loss

    def compute_train_val_loss_with_label_smoothing(self, batch, loss_reduction='mean'):
        # src, src_mask, trg, trg_mask = data
        nl_seqs, nl_char_seqs, ast_seqs, edge_paths_seqs = batch

        # out = self.forward(src, src_mask, trg, trg_mask)
        # predict=out[:-1]
        log_probs = self(nl_seqs, nl_char_seqs, ast_seqs[:-1, :], edge_paths_seqs[:-1, :, :])

        # predict = predict.view(-1, self.model.tgt_vocab_size)
        output_dim = log_probs.shape[-1]
        log_probs = log_probs.view(-1, output_dim)
        # target=trg[1:] und target = target.view(-1, 1)
        ast_seqs = ast_seqs[1:, :].view(-1, 1)

        # nll_loss = F.nll_loss(predict, target.view(-1), ignore_index=PAD_IDX)
        # loss = F.nll_loss(log_probs, ast_seqs, ignore_index=self.model.vocab_pad_id, reduction=loss_reduction)

        non_pad_mask = ast_seqs.ne(self.model.vocab_pad_id)
        nll_loss = -log_probs.gather(dim=-1, index=ast_seqs)[non_pad_mask].mean()
        smooth_loss = -log_probs.sum(dim=-1, keepdim=True)[non_pad_mask].mean()
        smooth_loss = smooth_loss / self.model.vocabtgt_size
        loss = (1. - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss

        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        train_perplexity_avg_loss = torch.exp(avg_loss)

        log.info("EPOCH %d RANK %d AVG TRAIN LOSS (cross entropy): %.5f ; AVG TRAIN PERPLEXITY: %.5f",
                 self.current_epoch, self.trainer.global_rank, avg_loss, train_perplexity_avg_loss)

        self.log('train_loss', avg_loss)

    def validation_step(self, batch, batch_idx):
        val_loss = self.compute_train_val_loss(batch)
        val_perplexity = torch.exp(val_loss)

        return {'val_loss': val_loss, 'val_perplexity': val_perplexity}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_perplexity_avg_loss = torch.exp(avg_loss)

        log.info("EPOCH %d Rank %d AVG VALIDATE LOSS (cross entropy): %.12f ; AVG VALIDATE PERPLEXITY: %.12f",
                 self.current_epoch, self.trainer.global_rank, avg_loss, val_perplexity_avg_loss)

        self.log('val_loss', avg_loss)
        self.log('val_perplexity_avg_loss', val_perplexity_avg_loss)

    def test_step(self, batch, batch_idx):
        log.info("CURRENT TEST BATCH #%d", batch_idx)
        nl_seqs, nl_char_seqs, snippets, ids, ast_seqs = batch

        predicted_codesnippets = []
        predicted_ast_seqs = []
        predicted_ast_seqs_enc = []

        i = 0
        while i < nl_seqs.size(1):
            nl_seq = nl_seqs[:, i]
            nl_seq = nl_seq.view(-1, 1)

            if not self.model.withcharemb:
                nl_char_seq = None
            else:
                nl_char_seq = nl_char_seqs[:, i]
                nl_char_seq = nl_char_seq.view(-1, 1, self.model.max_charseq_len)

            results = self.beamsearch.perform_beam_search(nl_seq, nl_char_seq)

            if results:
                ast_seq_lists, _, code_snippets = list(zip(*results))
            else:
                ast_seq_lists, code_snippets = [], []

            predicted_codesnippets.append(code_snippets)
            predicted_ast_seqs.append(
                [self.ast_seq_generator.visit_graph(ast_seq_list) for ast_seq_list, score, code_snippet in results])
            predicted_ast_seqs_enc.append(ast_seq_lists)

            i += 1

        return {'expected_codesnippet': snippets, 'predicted_codesnippets': predicted_codesnippets, 'ids': ids,
                'expected_ast_seq': ast_seqs, 'predicted_ast_seqs': predicted_ast_seqs,
                'predicted_ast_seqs_enc': predicted_ast_seqs_enc}

    def test_epoch_end(self, outputs):
        output_df = pd.DataFrame()

        for output in outputs:
            df = pd.DataFrame()

            df['expected_codesnippet'] = pd.Series(output['expected_codesnippet'])
            df['predicted_codesnippets'] = pd.Series(output['predicted_codesnippets'])
            df['id'] = pd.Series(output['ids'])
            df['expected_ast_seq'] = pd.Series(output['expected_ast_seq'])
            df['predicted_ast_seqs'] = pd.Series(output['predicted_ast_seqs'])
            df['predicted_ast_seqs_enc'] = pd.Series(output['predicted_ast_seqs_enc'])

            output_df = output_df.append(df)

        output_df.to_json(self.target_output_file, orient='records', lines=True)

    def collate_fn_train(self, batch):
        batch_size = len(batch)

        max_src_len = max([len(i[0]) for i in batch])
        src_padded = torch.zeros(max_src_len, batch_size, dtype=torch.long)

        max_tgt_len = max([len(i[2]) for i in batch])
        tgt_padded = torch.zeros(max_tgt_len, batch_size, dtype=torch.long)

        if self.model.withcharemb:
            char_padded = torch.zeros(max_src_len, batch_size, self.model.max_charseq_len, dtype=torch.long)
        else:
            char_padded = None
        edge_path_seqs_padded = torch.zeros(max_tgt_len, batch_size, self.model.max_path_depth, dtype=torch.long)

        for i, (src, char, tgt, edge_path_seqs, allowed_tokens) in enumerate(batch):
            src_size = len(src)
            src_padded[:src_size, i] = torch.tensor(src)

            if self.model.withcharemb:
                cseq_padded = torch.zeros(len(char), self.model.max_charseq_len, dtype=torch.long)

                for j, cseq in enumerate(char):
                    cseq_len = len(cseq)
                    if cseq_len > self.model.max_charseq_len:
                        cseq = cseq[:self.model.max_charseq_len]
                        cseq_len = self.model.max_charseq_len

                    cseq_padded[j, :cseq_len] = torch.as_tensor(cseq, dtype=torch.long)

                char_padded[:len(char), i, :self.model.max_charseq_len] = cseq_padded

            tgt_size = len(tgt)
            tgt_padded[:tgt_size, i] = torch.tensor(tgt)
            edge_path_seqs_padded[:tgt_size, i, :] = edge_path_seqs

        return src_padded, char_padded, tgt_padded, edge_path_seqs_padded

    def collate_fn_test(self, batch):
        batch_size = len(batch)
        max_src_len = max([len(i[0]) for i in batch])
        src_padded = torch.LongTensor(max_src_len, batch_size)
        src_padded.zero_()

        tgt_list = []
        id_list = []
        ast_seq_list = []

        if self.model.withcharemb:
            char_padded = torch.zeros(max_src_len, batch_size, self.model.max_charseq_len, dtype=torch.long)
        else:
            char_padded = None

        for i, (src, char, tgt, idx, ast_seq) in enumerate(batch):
            src_size = len(src)
            src_padded[:src_size, i] = torch.tensor(src)

            if self.model.withcharemb:
                cseq_padded = torch.zeros(len(char), self.model.max_charseq_len, dtype=torch.long)

                for j, cseq in enumerate(char):
                    cseq_len = len(cseq)
                    if cseq_len > self.model.max_charseq_len:
                        cseq = cseq[:self.model.max_charseq_len]
                        cseq_len = self.model.max_charseq_len

                    cseq_padded[j, :cseq_len] = torch.as_tensor(cseq, dtype=torch.long)

                char_padded[:len(char), i, :self.model.max_charseq_len] = cseq_padded

            tgt_list.append(tgt)
            id_list.append(idx)
            ast_seq_list.append(ast_seq)

        return src_padded, char_padded, tgt_list, id_list, ast_seq_list

    def configure_optimizers(self):
        optimzer = Adam(self.parameters(), lr=self.learning_rate, betas=(self.adam_beta_1, self.adam_beta_2),
                        eps=self.adam_epsilon)

        if self.scheduler == None:
            return optimzer

        elif self.scheduler == 'steplr':
            scheduler = StepLR(optimzer, step_size=self.step_size, gamma=self.gamma, last_epoch=self.sc_last_epoch)
        elif self.scheduler == 'lambdalr':
            lambda_lr = lambda epoch: self.d_model ** (-0.5) * min((epoch + 1) ** (-0.5),
                                                                   (epoch + 1) * self.warmup_steps ** (
                                                                       -1.5)) / self.learning_rate
            scheduler = LambdaLR(optimzer, lr_lambda=lambda_lr, last_epoch=self.sc_last_epoch)

        return [optimzer], [scheduler]


class LegacyTransformer(Transformer, ModelIO):
    """
        This class allows for loading models trained with many older versions
    """

    legacy_arg_names = set((
        "learning_rate", "batch_size", "num_dataloader_workers", "train_valid_data_path", "train_split",
        "val_split", "max_src_sentence_length", "max_tgt_sentence_length", "label_smoothing",
        "logits_forbidden_token_modifier_learning_rate", "scheduler", "step_size", "gamma", "warmup_steps",
        "sc_last_epoch", "adam_beta_1", "adam_beta_2", "adam_epsilon", "test_data_path", "grammar_graph_file",
        "target_language", "num_beams", "disable_decoder_constraint_mask", "max_beam_length",
        "max_num_predicted_results", "beam_search_mode", "keep_invalid_beamsearch_results", "validate_parsability",
        "target_output_file", "is_test_only_run"
    ))

    def __init__(self, *args, **kwargs):
        hparams = self.filter_non_legacy_args(kwargs)
        legacy_hparams = LegacyTransformer.filter_legacy_args(kwargs)

        super().__init__(*args, **hparams)
        self.hparams = hparams
        self.legacy_hparams = legacy_hparams

    @staticmethod
    def filter_legacy_args(d: Dict[str, object]):
        return {name: value for (name, value) in d.items() if name in LegacyTransformer.legacy_arg_names}

    @staticmethod
    def filter_non_legacy_args(d: Dict[str, object]):
        return {name: value for (name, value) in d.items() if name not in LegacyTransformer.legacy_arg_names}


def load_checkpoint(model_path, test_args):
    try:
        system = NL2CodeSystem.load_from_checkpoint(model_path, **test_args)
    except:
        legacy_test_args = LegacyTransformer.filter_legacy_args(test_args)
        model = LegacyTransformer.load_from_checkpoint(model_path, **legacy_test_args)
        legacy_train_args = {
            'learning_rate': model.legacy_hparams['learning_rate'],
        }
        legacy_train_data_args = {
            'train_valid_data_path': model.legacy_hparams['train_valid_data_path'],
            'train_split': model.legacy_hparams['train_split'],
            'val_split': model.legacy_hparams['val_split'],
            'max_src_sentence_length': model.legacy_hparams['max_src_sentence_length'],
            'max_tgt_sentence_length': model.legacy_hparams['max_tgt_sentence_length'],
        }
        system = NL2CodeSystem(**test_args, **legacy_train_args, model_hparams=model.hparams,
                               train_data_args=legacy_train_data_args)
        system.model = model

    if system.max_beam_length is None:
        system.max_beam_length = round(1.1 * int(system.train_data_args['max_tgt_sentence_length']))

    return system
