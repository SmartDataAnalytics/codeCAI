import copy

import torch.nn.functional as F
from torch.nn import Module, MultiheadAttention, Linear, Dropout, LayerNorm, ModuleList


class TransformerDecoder(Module):
    __constants__ = ['norm']

    def __init__(self,
                 decoder_layer,
                 num_layers,
                 att_layer=-1,
                 norm=None):

        super(TransformerDecoder, self).__init__()

        self.layers = ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.att_layer = (num_layers - 1 if att_layer == -1 else att_layer)

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        output = tgt

        for mod_counter, mod in enumerate(self.layers):
            if self.att_layer == mod_counter:
                output, cross_attention_weights = mod(output,
                                                memory,
                                                tgt_mask=tgt_mask,
                                                memory_mask=memory_mask,
                                                tgt_key_padding_mask=tgt_key_padding_mask,
                                                memory_key_padding_mask=memory_key_padding_mask)
            else:
                output, _ = mod(output,
                                memory,
                                tgt_mask=tgt_mask,
                                memory_mask=memory_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, cross_attention_weights


class TransformerDecoderLayer(Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 multihead_attention_dropout=0.0,
                 normalize_before=False,
                 activation="relu"):

        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(d_model,
                                            nhead,
                                            dropout=dropout)

        self.multihead_attn = MultiheadAttention(d_model,
                                                 nhead,
                                                 dropout=multihead_attention_dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu

        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        if self.normalize_before:
            tgt = self.norm1(tgt)

        tgt2 = self.self_attn(tgt,
                              tgt,
                              tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)

        if not self.normalize_before:
            tgt = self.norm1(tgt)
        else:
            tgt = self.norm2(tgt)

        tgt2, cross_attention_weights = self.multihead_attn(tgt,
                                                      memory,
                                                      memory,
                                                      attn_mask=memory_mask,
                                                      key_padding_mask=memory_key_padding_mask)

        tgt = tgt + self.dropout2(tgt2)

        if not self.normalize_before:
            tgt = self.norm2(tgt)
        else:
            tgt = self.norm3(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        tgt = tgt + self.dropout3(tgt2)

        if not self.normalize_before:
            tgt = self.norm3(tgt)

        return tgt, cross_attention_weights


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
