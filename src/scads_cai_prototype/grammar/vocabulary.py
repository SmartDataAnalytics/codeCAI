import io
from os.path import isfile

import torch
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer


class VocabularyCreator:
    def __init__(self, data, vocab_file=None, vocab_size=1000, vocab_type='bpe', unk_label='<unk>',
                 max_sentencepiece_length=16, split_by_whitespace=True, split_digits=False,
                 treat_whitespace_as_suffix=False, byte_fallback=False, add_dummy_prefix=True, user_defined_symbols=''):
        self.data = data
        self.vocab_file = vocab_file
        self.vocab_size = vocab_size
        self.vocab_type = vocab_type
        self.unk_label = unk_label
        self.max_sentencepiece_length = max_sentencepiece_length
        self.split_by_whitespace = split_by_whitespace
        self.split_digits = split_digits
        self.treat_whitespace_as_suffix = treat_whitespace_as_suffix
        self.byte_fallback = byte_fallback
        self.add_dummy_prefix = add_dummy_prefix
        self.user_defined_symbols = user_defined_symbols

    def create_vocab(self):
        if (isfile(self.vocab_file)):
            return False

        model = io.BytesIO()
        iter = self.data.__iter__()
        try:
            if self.vocab_type in ['word', 'char']:
                use_all_vocab = True
                size = 999999
            else:
                use_all_vocab = False
                size = self.vocab_size

            SentencePieceTrainer.train(sentence_iterator=iter,
                                       model_writer=model,
                                       vocab_size=size,
                                       num_threads=64,
                                       max_sentence_length=2 ** 20,  # max. 1 MB
                                       pad_id=0,
                                       unk_id=1,
                                       bos_id=2,
                                       eos_id=3,
                                       unk_piece=self.unk_label,
                                       unk_surface=self.unk_label,
                                       model_type=self.vocab_type,
                                       character_coverage=1.0,
                                       use_all_vocab=use_all_vocab,
                                       normalization_rule_name='nfkc',
                                       # add_dummy_prefix caused segmentation fault in bpe_model_trainer.cc
                                       # function Trainer::UpdateActiveSymbols()
                                       add_dummy_prefix=self.add_dummy_prefix,
                                       train_extremely_large_corpus=True,
                                       remove_extra_whitespaces=False,
                                       max_sentencepiece_length=self.max_sentencepiece_length,
                                       split_by_whitespace=self.split_by_whitespace,
                                       split_digits=self.split_digits,
                                       treat_whitespace_as_suffix=self.treat_whitespace_as_suffix,
                                       user_defined_symbols=self.user_defined_symbols
                                       )
        except:
            raise

        with open(self.vocab_file, 'wb') as f:
            f.write(model.getvalue())
        return True


class VocabularyLoader:
    def __init__(self, vocab_file=None):
        self.vocab_file = vocab_file

    def load_bpe_model(self):
        with open(self.vocab_file, 'rb') as f:
            sp = SentencePieceProcessor(model_proto=f.read())

            print('Loading vocabulary:', str(self.vocab_file), 'Vocabulary size:', len(sp))
            # print({id: sp.IdToPiece(id) for id in range(0, sp.vocab_size())})

        return sp


class VocabNLTransform:
    def __init__(self, vocab, add_bos=True, add_eos=True):
        self.vocab = vocab
        self.add_bos = add_bos
        self.add_eos = add_eos

    def __call__(self, sample):
        vocab_ids = self.vocab.encode(sample, add_bos=self.add_bos, add_eos=self.add_eos)

        return torch.LongTensor(vocab_ids)


class VocabAstSeqTransform:
    def __init__(self, vocab, add_bos=True, add_eos=True):
        self.vocab = vocab
        self.add_bos = add_bos
        self.add_eos = add_eos

    def __call__(self, sample):
        vocab_ids = self.vocab.encode(sample, add_bos=self.add_bos, add_eos=self.add_eos)
        tgt = torch.LongTensor(vocab_ids)
        return tgt
