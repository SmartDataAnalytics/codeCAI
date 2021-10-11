import logging as log
from typing import Optional

import torch
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader, random_split, Subset

from scads_cai_prototype.dataprovider import DataProvider
from scads_cai_prototype.dataset import TrainDataset, TestDataset, EdgePathTransform


class NL2CodeTrainDataModule(LightningDataModule):
    def __init__(self,
                 batch_size,
                 num_dataloader_workers,
                 max_src_sentence_length,
                 max_tgt_sentence_length,
                 max_path_depth,
                 path_multiple,
                 train_valid_data_path=None,
                 train_split=None,
                 val_split=None,
                 train_data_path=None,
                 valid_data_path=None,
                 max_charseq_len=None
                 ):
        super().__init__()

        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers

        self.train_valid_data_path = train_valid_data_path
        self.train_split = train_split
        self.val_split = val_split

        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path

        self.max_src_sentence_length = max_src_sentence_length
        self.max_tgt_sentence_length = max_tgt_sentence_length

        self.max_path_depth = max_path_depth
        self.path_multiple = path_multiple
        self.max_charseq_len = max_charseq_len
        self.withcharemb = max_charseq_len is not None

        self.train_dataset = None
        self.valid_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            column_default_values = {'nl_char_enc': None} if not self.withcharemb else {}

            if self.train_valid_data_path is not None:
                train_valid_dataprovider = DataProvider(self.train_valid_data_path)

                train_valid_data = train_valid_dataprovider.provide_data(
                    columns=['nl_enc', 'nl_char_enc', 'ast_seq_enc', 'edge_order_seq', 'allowed_tokens'],
                    column_default_values=column_default_values
                )

                train_valid_data['allowed_tokens'] = train_valid_data['allowed_tokens'].apply(
                    lambda allowed_tokens_list: [
                        torch.tensor(allowed_tokens, dtype=torch.long) for allowed_tokens in allowed_tokens_list])

                train_valid_dataset = TrainDataset(train_valid_data,
                                                   edge_path_transform=EdgePathTransform(self.max_path_depth,
                                                                                         self.path_multiple))

                train_s = round(self.train_split * len(train_valid_dataset))
                valid_s = len(train_valid_dataset) - train_s

                self.train_dataset, self.valid_dataset = random_split(train_valid_dataset, [train_s, valid_s])
                log.debug("Validation indices: %s", self.valid_dataset.indices)
            elif self.train_data_path is not None and self.valid_data_path is not None:
                train_dataprovider = DataProvider(self.train_data_path)

                train_data = train_dataprovider.provide_data(
                    columns=['nl_enc', 'nl_char_enc', 'ast_seq_enc', 'edge_order_seq', 'allowed_tokens'],
                    column_default_values=column_default_values
                )

                train_data['allowed_tokens'] = train_data['allowed_tokens'].apply(
                    lambda allowed_tokens_list: [
                        torch.tensor(allowed_tokens, dtype=torch.long) for allowed_tokens in allowed_tokens_list])

                self.train_dataset = TrainDataset(train_data, edge_path_transform=EdgePathTransform(self.max_path_depth,
                                                                                                    self.path_multiple))

                val_dataprovider = DataProvider(self.valid_data_path)

                val_data = val_dataprovider.provide_data(
                    columns=['nl_enc', 'nl_char_enc', 'ast_seq_enc', 'edge_order_seq', 'allowed_tokens'],
                    column_default_values=column_default_values
                )

                val_data['allowed_tokens'] = val_data['allowed_tokens'].apply(
                    lambda allowed_tokens_list: [
                        torch.tensor(allowed_tokens, dtype=torch.long) for allowed_tokens in allowed_tokens_list])
                self.valid_dataset = TrainDataset(val_data, edge_path_transform=EdgePathTransform(self.max_path_depth,
                                                                                                  self.path_multiple))
            else:
                raise ValueError("Either train-val data and train split or train data and val data must be specified")

    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self.get_dataloader(self.valid_dataset)

    def get_dataloader(self, dataset):
        retained_indices = [i for (i, (src, char, tgt, edge_order_seq, allowed_tokens)) in enumerate(dataset) if
                            len(src) <= self.max_src_sentence_length and len(tgt) <= self.max_tgt_sentence_length]

        subset = Subset(dataset, retained_indices)

        return DataLoader(subset, batch_size=self.batch_size, collate_fn=self.collate_fn, drop_last=False,
                          pin_memory=True, num_workers=self.num_dataloader_workers)

    def collate_fn(self, batch):
        batch_size = len(batch)

        max_src_len = max([len(i[0]) for i in batch])
        src_padded = torch.zeros(max_src_len, batch_size, dtype=torch.long)

        max_tgt_len = max([len(i[2]) for i in batch])
        tgt_padded = torch.zeros(max_tgt_len, batch_size, dtype=torch.long)

        if self.withcharemb:
            char_padded = torch.zeros(max_src_len, batch_size, self.max_charseq_len, dtype=torch.long)
        else:
            char_padded = None
        edge_path_seqs_padded = torch.zeros(max_tgt_len, batch_size, self.max_path_depth, dtype=torch.long)

        for i, (src, char, tgt, edge_path_seqs, allowed_tokens) in enumerate(batch):
            src_size = len(src)
            src_padded[:src_size, i] = torch.tensor(src)

            if self.withcharemb:
                cseq_padded = torch.zeros(len(char), self.max_charseq_len, dtype=torch.long)

                for j, cseq in enumerate(char):
                    cseq_len = len(cseq)
                    if cseq_len > self.max_charseq_len:
                        cseq = cseq[:self.max_charseq_len]
                        cseq_len = self.max_charseq_len

                    cseq_padded[j, :cseq_len] = torch.as_tensor(cseq, dtype=torch.long)

                char_padded[:len(char), i, :self.max_charseq_len] = cseq_padded

            tgt_size = len(tgt)
            tgt_padded[:tgt_size, i] = torch.tensor(tgt)
            edge_path_seqs_padded[:tgt_size, i, :] = edge_path_seqs

        return src_padded, char_padded, tgt_padded, edge_path_seqs_padded


class NL2CodeTestDataModule(LightningDataModule):
    def __init__(self,
                 batch_size,
                 num_dataloader_workers,
                 test_data_path,
                 is_test_only_run,
                 max_charseq_len=None,
                 ):
        super().__init__()

        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.test_data_path = test_data_path
        self.is_test_only_run = is_test_only_run
        self.test_dataprovider = DataProvider(self.test_data_path)

        self.max_charseq_len = max_charseq_len
        self.withcharemb = self.max_charseq_len is not None

    def test_dataloader(self):
        column_default_values = {'nl_char_enc': None} if not self.withcharemb else {}
        test_data = self.test_dataprovider.provide_data(columns=['nl_enc', 'nl_char_enc', 'snippet'],
                                                        column_default_values=column_default_values)
        test_dataset = TestDataset(test_data)

        # FIXME: Batch size != 1 doesn't work properly
        batch_size = 1
        return DataLoader(test_dataset, batch_size=batch_size, collate_fn=self.collate_fn, drop_last=False,
                          pin_memory=True, num_workers=self.num_dataloader_workers)

    def collate_fn(self, batch):
        batch_size = len(batch)
        max_src_len = max([len(i[0]) for i in batch])
        src_padded = torch.LongTensor(max_src_len, batch_size)
        src_padded.zero_()

        tgt_list = []
        id_list = []
        ast_seq_list = []

        if self.withcharemb:
            char_padded = torch.zeros(max_src_len, batch_size, self.max_charseq_len, dtype=torch.long)
        else:
            char_padded = None

        for i, (src, char, tgt, idx, ast_seq) in enumerate(batch):
            src_size = len(src)
            src_padded[:src_size, i] = torch.tensor(src)

            if self.withcharemb:
                cseq_padded = torch.zeros(len(char), self.max_charseq_len, dtype=torch.long)

                for j, cseq in enumerate(char):
                    cseq_len = len(cseq)
                    if cseq_len > self.max_charseq_len:
                        cseq = cseq[:self.max_charseq_len]
                        cseq_len = self.max_charseq_len

                    cseq_padded[j, :cseq_len] = torch.as_tensor(cseq, dtype=torch.long)

                char_padded[:len(char), i, :self.max_charseq_len] = cseq_padded

            tgt_list.append(tgt)
            id_list.append(idx)
            ast_seq_list.append(ast_seq)

        return src_padded, char_padded, tgt_list, id_list, ast_seq_list
