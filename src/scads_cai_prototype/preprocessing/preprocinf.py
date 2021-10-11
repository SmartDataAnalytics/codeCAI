import argparse
import logging as log
import os

import torch
from sentencepiece import SentencePieceProcessor

from scads_cai_prototype.grammar.vocabulary import VocabularyLoader


class Preprocinf():
    def __init__(self, vocabsrc_file, vocabchar_file=None):
        self.vocab_src: SentencePieceProcessor = self.load_vocab(vocabsrc_file)
        self.vocab_char: SentencePieceProcessor = self.load_vocab(vocabchar_file) if vocabchar_file else None

    def load_vocab(self, vocab_file):
        if not os.path.exists(vocab_file):
            raise ValueError('No vocabulary found ' + vocab_file)
        else:
            print('Loading source vocabulary', vocab_file)
            return VocabularyLoader(vocab_file=vocab_file).load_bpe_model()

    def preproc(self, nl):
        return tuple(self.vocab_src.encode(nl, add_bos=True, add_eos=True))

    def preproc_char_seq(self, nl):
        nl_dec = self.vocab_src.EncodeAsPieces(nl)
        nl_dec.append('')
        nl_dec.insert(0, '')
        char_seq_enc = [tuple(self.vocab_char.encode(t, add_bos=False, add_eos=False)) for t in nl_dec]

        return char_seq_enc

    def pad_char_seq(self, char_seqs, max_char_seq_len):
        cseq_padded = torch.zeros(len(char_seqs), max_char_seq_len, dtype=torch.long)

        for i, cseq in enumerate(char_seqs):
            cseq_len = len(cseq)

            if cseq_len > max_char_seq_len:
                cseq = cseq[:max_char_seq_len]
                cseq_len = max_char_seq_len

            cseq_padded[i, :cseq_len] = torch.as_tensor(cseq, dtype=torch.long)

        return cseq_padded


def get_args():
    parser = argparse.ArgumentParser("Preproc data inference", fromfile_prefix_chars='@')
    parser.add_argument("--vocabsrc-file", type=str, default='vocabsrc.model')
    parser.add_argument("--vocabchar-file", type=str, default=None)
    parser.add_argument("--nl-input", type=str, default='')
    return parser.parse_args()


def main():
    args = get_args()
    print("Preproc inference args:", vars(args))
    log.basicConfig(level=log.DEBUG)

    preproc = Preprocinf(vocabsrc_file=args.vocabsrc_file, vocabchar_file=args.vocabchar_file)
    preproc.preproc(args.nl_input)


if __name__ == '__main__':
    main()
