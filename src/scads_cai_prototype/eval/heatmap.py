import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns


def get_args():
    parser = argparse.ArgumentParser("Attention Weights", fromfile_prefix_chars='@')

    # Attention weights logging
    parser.add_argument("--self-attention-weights-file", type=str, default=None)
    parser.add_argument("--cross-attention-weights-file", type=str, default=None)
    parser.add_argument("--nl-ast-seq-file", type=str, default=None)

    return parser.parse_args()

def plotcrossattheatmap(cols, rows, data):
    fig, ax = plt.subplots(figsize=(5, 8))
    plt.subplots_adjust(left=0.2, right=0.8, top=0.87, bottom=0.05)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)
    ax.tick_params(labelsize=12)
    hm=sns.heatmap(data,xticklabels=ax.get_xticklabels(), yticklabels=ax.get_yticklabels(), ax=ax, cmap="Blues")
    hm.set_xticklabels(cols, rotation=30, ha="left", rotation_mode="anchor")
    hm.set_yticklabels(rows, rotation=0)
    hm.xaxis.set_ticks_position('top')
    plt.show()

def plotselfattheatmap(cols, rows, data):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)
    ax.tick_params(labelsize=12)
    hm=sns.heatmap(data,xticklabels=ax.get_xticklabels(), yticklabels=ax.get_yticklabels(), ax=ax, cmap="Blues")
    hm.set_xticklabels(cols, rotation=90)
    hm.set_yticklabels(rows, rotation=0)
    hm.xaxis.set_ticks_position('top')
    plt.show()


def plotheatmap(cols, rows, data):
    fig, ax = plt.subplots()
    im = ax.imshow(data)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    # ... and label them with the respective list entries.
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="left",
             rotation_mode="anchor")

    fig.tight_layout()
    plt.show()


def read_from_file(file):
    with open(file, 'r') as f:
        return json.load(f)


def main():
    args = get_args()

    nl_ast_seqs = read_from_file(args.nl_ast_seq_file)
    nl_input_seq = nl_ast_seqs['nl_seq_dec']
    ast_output_seq = nl_ast_seqs['ast_seq_dec']

    cross_attention_weights = read_from_file(args.cross_attention_weights_file)
    self_attention_weights = read_from_file(args.self_attention_weights_file)

    #plotheatmap(nl_input_seq, ast_output_seq[1:], np.array(cross_attention_weights))
    #plotheatmap(ast_output_seq[:-1], ast_output_seq[1:], np.array(self_attention_weights))
    #plotsnsheatmap(ast_output_seq[:-1], ast_output_seq[1:], np.array(self_attention_weights))
    plotselfattheatmap(ast_output_seq[:-1], ast_output_seq[1:], np.array(self_attention_weights))
    plotcrossattheatmap(nl_input_seq, ast_output_seq[1:], np.array(cross_attention_weights))


if __name__ == '__main__':
    main()
