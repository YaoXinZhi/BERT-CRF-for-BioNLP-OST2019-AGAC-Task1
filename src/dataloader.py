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
                 unknown_token='[UNK]', only_labeled_data=True, only_loaded_label:bool=False,
                 o_label:str='O', use_word_piece:bool=True):

        self.data_path = data_path
        self.label_path = label_path
        self.vocab_dict = vocab_dict
        self.unknown_token = unknown_token
        self.only_labeled_data=only_labeled_data
        self.only_loaded_label = only_loaded_label
        self.o_label = o_label
        self.use_word_piece = use_word_piece

        self.data = []
        self.label = []

        self.label_set = set()
        self.label_to_index = {}
        self.index_to_label = {}

        self.read_label()
        #print(self.label_set)
        self.read_bio_data()

        if '' in self.data:
            self.data.remove('')

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

    def read_bio_data(self):
        data_list = []
        label_list = []

        with open(self.data_path, encoding='utf-8') as f:
            for line in f:
                l = line.strip().split('\t')
                if len(l) < 2:
                    if data_list and label_list:
                        if label_list[0].startswith('I-'):
                            logging.warning(f'Warning label prefix:')
                            logging.warning(data_list)
                            logging.warning(label_list)

                        if self.only_labeled_data:
                            if not any(label.startswith('B-') or label.startswith('I-') for label in label_list):
                                continue
                        # fixme: word piece
                        if not self.use_word_piece:
                            data_list = [str(self.vocab_dict[token]) for token in data_list]

                        self.data.append('&&&'.join(data_list))
                        self.label.append('&&&'.join(label_list))

                        data_list = []
                        label_list = []

                else:
                    token, label = l

                    if not self.use_word_piece:
                        if not self.vocab_dict.get(token):
                            token = self.unknown_token

                    if not (label.startswith('B-') or label.startswith('I-') or label == 'O'):
                        logging.warning('wrong label:')
                        logging.warning(f'{token}\t{label}')

                    data_list.append(token)
                    if self.only_loaded_label:
                        if label in self.label_set:
                            label_list.append(label)
                        else:
                            label_list.append(self.o_label)
                    else:
                        label_list.append(label)



class Infer_Dataset(Dataset):
    def __init__(self, data_path: str, label_path: str, vocab_dict: dict,
                 unknown_token='[UNK]', use_word_piece:bool=True):

        self.data_path = data_path
        self.label_path = label_path
        self.vocab_dict = vocab_dict
        self.unknown_token = unknown_token
        self.use_word_piece=use_word_piece
        self.data = []
        self.orig_data = []

        self.label_set = set()
        self.label_to_index = {}
        self.index_to_label = {}

        self.read_label()
        self.read_infer_data()
        logging.info(f'Data Path: {self.data_path},'
              f' Data size: {len(self.data):,} sentences,'
                     f'Original Data: {len(self.orig_data):,}')
        logging.info(f'Total label count: {len(self.label_set)}.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.orig_data[item]

    def read_label(self):
        with open(self.label_path) as f:
            for line in f:
                label = line.strip()
                self.label_set.add(label)

        self.label_to_index = {label: idx for idx, label in enumerate(self.label_set)}
        self.index_to_label = {idx: label for idx, label in enumerate(self.label_set)}

    def read_infer_data(self):
        data_list = []
        orig_data_list = []

        with open(self.data_path, encoding='utf-8') as f:
            for line in f:
                l = line.strip().split('\t')
                if l == ['']:

                    if not self.use_word_piece:
                        data_list = [str(self.vocab_dict[token]) for token in data_list]

                    self.data.append('&&&'.join(data_list))
                    self.orig_data.append('&&&'.join(orig_data_list))

                    data_list = []
                    orig_data_list = []
                else:
                    token = l[2]
                    orig_data_list.append('$$$'.join(l))

                    if not self.use_word_piece:
                        if not self.vocab_dict.get(token):
                            token = self.unknown_token

                    data_list.append(token)


if __name__ == '__main__':
    pass

    # from src.config import args
    # from transformers import BertTokenizerFast
    #
    # from torch.utils.data import DataLoader
    #
    # # paras = args()
    #
    # tokenizer = BertTokenizerFast.from_pretrained(paras.model_name)
    #
    # vocab_dict = tokenizer.get_vocab()
    # idx_to_word = {idx: vocab for vocab, idx in vocab_dict.items()}
    # # infer_dataset = Infer_Dataset(paras.infer_data, paras.label_file,
    # #                               vocab_dict)
    # #
    # # infer_dataloader = DataLoader(infer_dataset, paras.batch_size,
    # #                               shuffle=False, drop_last=False)
    # #
    # # for batch in infer_dataloader:
    # #     batch_data, batch_orig_data = batch
    # #
    # #     for i in range(paras.batch_size):
    # #         print(len(batch_data[i].split('&&&')), len(batch_orig_data[i].split('&&&')))
    # #
    # #     break
    # #
    # #
    # train_dataset = SeqLabeling_Dataset(paras.train_data, paras.label_file,
    #                                     vocab_dict, only_labeled_data=paras.load_labeled_data,
    #                                     only_loaded_label=paras.only_loaded_label,
    #                                     use_word_piece=paras.use_word_piece)
    # train_dataloader = DataLoader(train_dataset, batch_size=paras.batch_size,
    #                               shuffle=paras.shuffle, drop_last=paras.drop_last)
    # label_set = train_dataset.label_set
    # label_to_index = train_dataset.label_to_index
    #
    # # batch data
    # batch = ''
    # for batch in train_dataloader:
    #     break
    #
    # batch_data, batch_label = batch
    # batch_data_list = [ data.split('&&&') for data in batch_data ]
    # batch_label_list = [ label.split('&&&') for label in batch_label ]
    # # #
    # # # # no wordpiece
    # # input_ids, mask = batch_data_processing(batch_data_list, paras.max_length,
    # #                                         vocab_dict.get('[PAD]'),
    # #                                         vocab_dict.get('[CLS]'),
    # #                                         vocab_dict.get('[SEP]'))
    # # input_ids = input_ids
    # # mask = mask
    # #
    # # batch_max_length = input_ids.shape[ 1 ]
    # # batch_label_pad = label_padding(paras.max_length, batch_max_length, batch_label_list,
    # #                                 label_to_index)
    # #
    # # show_example(input_ids, batch_label_list, tokenizer)
    # # # torch.Size([4, 35]) torch.Size([4, 35])
    # # print(input_ids.shape, batch_label_pad.shape)
    # #
    # # # wordpiece
    # input_ids_wp, batch_mask_wp, batch_adjust_label = batch_data_wordpiece_processing(tokenizer, batch_data_list,
    #                                                                                   paras.max_length, batch_label_list)
    # # torch.Size([ 4, 72 ])
    # # torch.Size([ 4, 72 ])
    # print(input_ids_wp.shape, batch_mask_wp.shape)
    #
    # ## decoded see
    # batch_decoded_list = convert_index_to_token(input_ids_wp, tokenizer)
    #
    # for ids_list, adjust_label in zip(input_ids_wp, batch_adjust_label):
    #     for _id, label in zip(ids_list, adjust_label):
    #         print(idx_to_word[int(_id)], label)
    #
    # batch_length = input_ids_wp.shape[1]
    # # 4* 72
    # batch_label_pad_wp = label_padding_with_special_token(batch_length, batch_adjust_label,
    #                                                       label_to_index)
    #
    #
    # show_example(input_ids_wp, batch_adjust_label, tokenizer)
