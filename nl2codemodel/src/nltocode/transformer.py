import logging as log
import math

import torch
import torch.nn as nn
from torch.nn import Parameter, Module

import nltocode.decoder as custom_decoder
import nltocode.encoder as custom_encoder


class Transformer(Module):
    def __init__(self,
                 d_model,
                 nhead,
                 num_encoder_layers,
                 num_decoder_layers,
                 dim_feedforward,
                 dropout,
                 multihead_attention_dropout,
                 normalize_before,
                 activation,
                 vocabsrc_size,
                 vocabtgt_size,
                 vocab_pad_id,
                 max_path_depth,
                 path_multiple=0,
                 tgt_pos_enc_type=None,
                 logits_forbidden_token_modifier=None,
                 logits_forbidden_token_modifier_schedule=None,
                 logits_forbidden_token_op=None,
                 enable_copy=True,
                 copy_att_layer=-1,
                 withcharemb=False,
                 vocabchar_size=None,
                 max_charseq_len=None,
                 share_nl_weights=False,
                 ):
        super().__init__()

        if logits_forbidden_token_modifier is not None:
            raise ValueError("Forbidden token modifier support has been removed")

        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.multihead_attention_dropout = multihead_attention_dropout
        self.normalize_before = normalize_before
        self.activation = activation

        self.withcharemb = withcharemb

        self.vocabsrc_size = vocabsrc_size
        self.vocabtgt_size = vocabtgt_size
        self.vocab_pad_id = vocab_pad_id

        self.vocabchar_size = vocabchar_size
        self.max_charseq_len = max_charseq_len

        self.tgt_pos_enc_type = tgt_pos_enc_type

        self.max_path_depth = max_path_depth
        self.path_multiple = path_multiple

        self.enable_copy = enable_copy
        self.share_nl_weights = share_nl_weights

        self.encoder_src = nn.Embedding(self.vocabsrc_size, self.d_model)
        self.encoder_tgt = nn.Embedding(self.vocabtgt_size, self.d_model)

        if self.withcharemb:
            self.encoder_char = nn.Embedding(self.vocabchar_size, self.d_model)
            self.linear_char = nn.Linear(self.d_model * self.max_charseq_len, self.d_model)
            self.norm_char = nn.LayerNorm(self.d_model)

        if self.share_nl_weights:
            # According to:
            # https://discuss.pytorch.org/t/how-to-use-shared-weights-in-different-layers-of-a-model/71263/2
            del self.encoder_src.weight
            del self.encoder_tgt.weight
            self.src_weights = Parameter(torch.Tensor(vocabsrc_size, self.d_model))
            self.tgt_weights = Parameter(torch.Tensor(vocabtgt_size - vocabsrc_size, self.d_model))
            torch.nn.init.normal_(self.src_weights)
            torch.nn.init.normal_(self.tgt_weights)
            self.encoder_src.weight = self.src_weights
            self.encoder_tgt.weight = torch.cat([self.src_weights, self.tgt_weights], dim=0)

        self.dropout_reg_src = nn.Dropout(p=self.dropout)
        self.dropout_reg_tgt = nn.Dropout(p=self.dropout)

        decoder_layer = custom_decoder.TransformerDecoderLayer(d_model=self.d_model,
                                                               nhead=self.nhead,
                                                               dim_feedforward=self.dim_feedforward,
                                                               dropout=self.dropout,
                                                               multihead_attention_dropout=self.multihead_attention_dropout,
                                                               normalize_before=self.normalize_before)

        self.decoder = custom_decoder.TransformerDecoder(decoder_layer,
                                                         num_layers=self.num_decoder_layers,
                                                         att_layer=copy_att_layer,
                                                         norm=nn.LayerNorm(self.d_model))

        encoder_layer = custom_encoder.TransformerEncoderLayer(d_model=self.d_model,
                                                               nhead=self.nhead,
                                                               dim_feedforward=self.dim_feedforward,
                                                               dropout=self.dropout,
                                                               activation=self.activation,
                                                               withcharemb=self.withcharemb)

        encoder_norm = nn.LayerNorm(self.d_model)

        self.encoder = custom_encoder.TransformerEncoder(encoder_layer,
                                                         self.num_encoder_layers,
                                                         encoder_norm)

        self.lin_vocab = nn.Linear(self.d_model, vocabtgt_size)

        if self.enable_copy:
            self.p_gen = nn.Sequential(nn.Linear(self.d_model * 3, 1), nn.Sigmoid())

        self.check_hyperparameters()

    def check_hyperparameters(self):
        if self.d_model % self.max_path_depth != 0:
            raise ValueError("d_model must be divisible by max_depth")

    def generate_square_subsequent_mask(self, m):
        sz = m.size(0)
        mask = (torch.triu(torch.ones(sz, sz, device=m.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, char, tgt, edge_path_seqs, src_encoding=None):
        '''
            Dimensions: S = src max length, T = tgt max length, B = batch size, C = char emb. size, P = max path depth

            src: Long tensor S x B
            char: Long tensor S x B x C
            tgt: Long tensor T x B
            edge_path_seqs: Long tensor T x B x P
            src_encoding: Optional (cached) return value of self.compute_src_encoding(src, char)
        '''

        try:
            if self.share_nl_weights:
                # Mostly according to:
                # https://discuss.pytorch.org/t/how-to-use-shared-weights-in-different-layers-of-a-model/71263/2
                self.encoder_src.weight = self.src_weights
                self.encoder_tgt.weight = torch.cat([self.src_weights, self.tgt_weights], dim=0)
                # self.encoder_tgt.weight = self.tgt_weights.clone()
                # self.encoder_tgt.weight[:self.vocabsrc_size, :] = self.src_weights

            enc_hs = self.encode_src(src, char) if src_encoding is None else src_encoding

            # TODO: generate_square_subsequent_mask(len(tgt)).to(tgt.device)
            mask = torch.triu(torch.ones(tgt.size(0), tgt.size(0)), 1)
            mask = mask.type_as(tgt).type(torch.float)
            tgt_mask = mask.masked_fill(mask == 1, float('-inf'))

            tgt_pad_mask = (tgt == 0).transpose(0, 1)
            tgt_embeddings = self.encoder_tgt(tgt)
            tgt_embeddings_with_pos = self.pos_encode(tgt_embeddings, self.tgt_pos_enc_type, edge_path_seqs)
            tgt_embeddings_with_pos = self.dropout_reg_tgt(tgt_embeddings_with_pos)
            dec_hs, attention = self.decoder(tgt_embeddings_with_pos,
                                             enc_hs,
                                             tgt_mask=tgt_mask,
                                             memory_mask=None,
                                             tgt_key_padding_mask=tgt_pad_mask,
                                             memory_key_padding_mask=None)

            logits = self.lin_vocab(dec_hs)

            if self.enable_copy:
                return self.apply_copy(attention, dec_hs, enc_hs, logits, src, tgt_embeddings_with_pos)
            else:
                return torch.log_softmax(logits, dim=-1)
        except:
            log.error("ERROR SRC/TGT SHAPE: %s / %s", src.shape, tgt.shape)
            raise

    def encode_src(self, src, char):
        src_pad_mask = (src == 0).transpose(0, 1)
        src_embeddings = self.encoder_src(src)
        src_embeddings_with_pos = self.pos_encode_seq(src_embeddings, self.d_model)
        src_embeddings_with_pos = self.dropout_reg_src(src_embeddings_with_pos)

        if self.withcharemb:
            char_embeddings = self.encoder_char(char)
            char_embeddings = char_embeddings.view(-1, char.size(1), self.d_model * self.max_charseq_len)
            char_embeddings = self.linear_char(char_embeddings)
            char_embeddings = self.norm_char(char_embeddings)
        else:
            char_embeddings = None

        enc_hs = self.encoder(src_embeddings_with_pos,
                              char_embeddings,
                              mask=None,
                              src_key_padding_mask=src_pad_mask)
        return enc_hs

    def apply_copy(self, attention, dec_hs, enc_hs, logits, src, tgt_embeddings_with_pos):
        p_vocab = logits.softmax(dim=-1)
        hidden_states = enc_hs.transpose(0, 1)
        context_vectors = torch.matmul(attention, hidden_states).transpose(0, 1)
        total_states = torch.cat((context_vectors, dec_hs, tgt_embeddings_with_pos), dim=-1)
        p_gen = self.p_gen(total_states)
        p_copy = 1 - p_gen
        src_t = src.transpose(0, 1)
        one_hot = torch.zeros(src_t.size(0),
                              src_t.size(1),
                              self.vocabsrc_size,
                              device=src_t.device)
        one_hot = one_hot.scatter_(dim=-1,
                                   index=src_t.unsqueeze(-1),
                                   value=1)
        p_copy_src_vocab = torch.matmul(attention, one_hot)
        input_vocab = torch.arange(self.vocabtgt_size - self.vocabsrc_size,
                                   self.vocabtgt_size,
                                   device=src_t.device)
        src_to_tgt_conversion_matrix = torch.zeros(self.vocabsrc_size,
                                                   self.vocabtgt_size,
                                                   device=src_t.device)
        src_to_tgt_conversion_matrix_scatter = src_to_tgt_conversion_matrix.scatter_(dim=-1,
                                                                                     index=input_vocab.unsqueeze(
                                                                                         -1), value=1)

        p_copy_tgt_vocab = torch.matmul(p_copy_src_vocab, src_to_tgt_conversion_matrix_scatter).transpose(0, 1)
        p = torch.add(p_vocab * p_gen, p_copy_tgt_vocab * p_copy)
        log_probs = torch.log(p)

        return log_probs

    def pos_encode(self, embeddings, pos_enc_type, edge_path_seqs):
        if pos_enc_type == 'tree':
            return self.pos_encode_tree(embeddings, edge_path_seqs, self.d_model)
        elif pos_enc_type == 'seq':
            return self.pos_encode_seq(embeddings, self.d_model)
        elif pos_enc_type == 'comb':
            return self.pos_encode_combined(embeddings, edge_path_seqs)
        elif pos_enc_type is None:
            raise ValueError("Positional encoding type not specified")
        else:
            raise ValueError('Unknown positional encoding type %s' % pos_enc_type)

    def pos_encode_seq(self, embeddings, dmodel):
        length = embeddings.size(0)
        pos_encoder = torch.zeros(length, dmodel).type_as(embeddings)
        pos = torch.arange(0, length, dtype=torch.float).type_as(embeddings).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dmodel, 2).type_as(embeddings) * (-math.log(10000.0) / dmodel))
        phase = pos * div_term
        pos_encoder[:, 0::2] = torch.sin(phase)
        pos_encoder[:, 1::2] = torch.cos(phase)
        pos_encoder = pos_encoder.unsqueeze(0).transpose(0, 1)
        embeddings_with_pos = embeddings + pos_encoder

        return embeddings_with_pos

    def pos_encode_tree(self, embeddings, edge_path_seqs, dmodel):
        max_sentence_length = embeddings.size(0)
        batch_size = embeddings.size(1)
        max_depth = edge_path_seqs.size(2)
        d_pos = dmodel // max_depth

        pos = edge_path_seqs.unsqueeze(-1)
        freq_index = torch.arange(0, d_pos, 2, device=embeddings.device, dtype=torch.float)
        frequency = torch.exp(freq_index * (-math.log(10000.0) / d_pos))
        phase = pos * frequency
        phase = phase.view(max_sentence_length, batch_size, max_depth * d_pos // 2)

        padding = (phase == 0)

        pos_encoder = torch.zeros(max_sentence_length, batch_size, max_depth * d_pos, device=embeddings.device)
        pos_encoder[:, :, 0::2] = torch.sin(phase)
        pos_encoder[:, :, 1::2] = torch.cos(phase)

        padding_replacement = 0.0
        pos_encoder[:, :, 0::2].masked_fill_(padding, padding_replacement)
        pos_encoder[:, :, 1::2].masked_fill_(padding, padding_replacement)

        embeddings_with_pos = embeddings + pos_encoder

        return embeddings_with_pos

    def pos_encode_combined(self, embeddings, edge_path_seqs):
        dmodel = self.d_model // 2
        seq_enc = self.pos_encode_seq(embeddings[:, :, dmodel:], dmodel)
        tree_enc = self.pos_encode_tree(embeddings[:, :, :dmodel], edge_path_seqs, dmodel)
        return torch.cat([tree_enc, seq_enc], dim=-1)
