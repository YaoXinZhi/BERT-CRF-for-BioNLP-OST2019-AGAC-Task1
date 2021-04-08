# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 31/03/2021 9:38
@Author: XINZHI YAO
"""

import logging

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)


class SeqLabeling_Dataset(Dataset):
    def __init__(self, data_path: str, label_path: str, vocab_dict: dict,
                 unknown_token='[UNK]'):

        self.data_path = data_path
        self.label_path = label_path
        self.vocab_dict = vocab_dict
        self.unknown_token = unknown_token
        self.data = []
        self.label = []

        self.label_set = set()
        self.label_to_index = {}
        self.index_to_label = {}

        self.read_label()
        self.read_data()
        logging.info(f'Data Path: {self.data_path},'
              f' Data size: {len(self.data):,} sentences,'
                     f' {len(self.label):,} labels.')
        logging.info(f'Total label count: {len(self.label_set)}.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        return self.data[item], self.label[item]

    def read_label(self):
        with open(self.label_path) as f:
            for line in f:
                label = line.strip()
                self.label_set.add(label)

        self.label_to_index = {label: idx for idx, label in enumerate(self.label_set)}
        self.index_to_label = {idx: label for idx, label in enumerate(self.label_set)}


    def read_data(self):
        data_list = []
        label_list = []

        with open(self.data_path, encoding='utf-8') as f:
            for line in f:
                l = line.strip().split('\t')
                if len(l) < 2:
                    if data_list and label_list:
                        # Warning label list
                        if label_list[0].startswith('I-'):
                            logging.warning(f'Warning label prefix:')
                            logging.warning(data_list)
                            logging.warning(label_list)

                        # fixme: Only load data with annotations
                        if not any(label.startswith('B-') or label.startswith('I-') for label in label_list):
                            continue

                        data_index_list = [str(self.vocab_dict[token]) for token in data_list]

                        # fixme: joined by &&&
                        # fixme: data_index_list
                        self.data.append('&&&'.join(data_index_list))
                        self.label.append('&&&'.join(label_list))

                        data_list = []
                        label_list = []

                else:
                    token, label = l

                    # fixme: [UNK] token replace
                    if not self.vocab_dict.get(token):
                        token = self.unknown_token
                        token_idx = self.vocab_dict.get(self.unknown_token)
                    else:
                        token_idx = self.vocab_dict.get(token)

                    if not (label.startswith('B-') or label.startswith('I-') or label == 'O'):
                        logging.warning('wrong label:')
                        logging.warning(f'{token}\t{label}')

                    data_list.append(token)
                    label_list.append(label)


if __name__ == '__main__':
    pass

    # train_file = '../data/train_input.txt'
    #
    # train_dataset = SeqLabeling_Dataset(train_file)
    #
    # train_dataloader = DataLoader(train_dataset, 3, shuffle=True, drop_last=False)
    #
    # tokenizer = BertTokenizerFast.from_pretrained(model_name)
    #
    # for idx, batch in enumerate(train_dataloader):
    #     # print(batch)
    #     batch_data, batch_label = batch
    #
    #     print(batch_label[0])
    #     if 'B-' in batch_label[0]:
    #         break
    #
    # batch_data_list = [data.split('&&&') for data in batch_data]
    # batch_label_list = [label.split('&&&') for label in batch_label]
    #
    #
    # # example
    # # batch_data_list = batch_data[0].split('&&&')
    # # batch_label_list = batch_label[0].split('&&&')
    #
    # # for i, j in zip(batch_data_list, batch_label_list):
    # #     print(i, j)
    #
    # max_seq_length=50
    #
    # tokens = tokenizer(batch_data_list,
    #                    return_offsets_mapping=True,
    #                    max_length=max_seq_length,
    #                    truncation=True,
    #                    is_split_into_words=True)
    #
    # batch_input_ids = tokens[ 'input_ids' ]
    # batch_token_type_ids = tokens[ 'token_type_ids' ]
    # batch_attention_mask = tokens[ 'attention_mask' ]
    # batch_offset_mapping = tokens[ 'offset_mapping' ]
    #
    # # for i, j in zip(batch_input_ids, batch_offset_mapping):
    # #     print(tokenizer.decode(i), j )
    #
    # batch_adjust_label_list = batch_adjust_label(batch_offset_mapping,
    #                                              batch_label_list)
    #
    # # adjust_label_list = adjust_label_by_offset(batch_offset_mapping, batch_label_list,
    # #                                            batch_input_ids)
    #
    # # len(adjust_label_list)
    # # len(batch_offset_mapping)


