import copy

import torch
import torch.nn.functional as F
from torch.nn import Module, MultiheadAttention, Linear, Dropout, LayerNorm, ModuleList


class TransformerEncoder(Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, char_emb=None, mask=None, src_key_padding_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, char_emb, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", withcharemb=False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.withcharemb = withcharemb

        if withcharemb:
            self.linear_q = Linear(d_model, d_model)
            self.linear_ky = Linear(d_model, d_model)
            self.linear_kc = Linear(d_model, d_model)
            self.linear_vy = Linear(d_model, d_model)
            self.linear_vc = Linear(d_model, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, y, c, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(y, y, y, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        y = y + self.dropout1(src2)
        y = self.norm1(y)

        # Char Embedding/Gating mechanism
        if self.withcharemb:
            q = self.linear_q(y)
            ky = self.linear_ky(y)
            kc = self.linear_kc(c)
            vy = self.linear_vy(y)
            vc = self.linear_vc(c)

            q_ky = torch.sum(q * ky, dim=2)
            q_kc = torch.sum(q * kc, dim=2)

            q_kyc = torch.stack((q_ky, q_kc), dim=2)
            a = q_kyc.softmax(dim=2)

            y = a[:, :, 0:1] * vy + a[:, :, 1:2] * vc

        src2 = self.linear2(self.dropout(self.activation(self.linear1(y))))
        y = y + self.dropout2(src2)
        y = self.norm2(y)
        return y


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])
