# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 31/03/2021 9:39
@Author: XINZHI YAO
"""

import os
import logging


import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import AdamW, WarmUp
from transformers import BertTokenizerFast, BertModel, BertTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)

# todo: tensorboardX
from tensorboardX import SummaryWriter

# from model import BERT_CRF
# from utils import *
# from dataloader import SeqLabeling_Dataset
# from config import args


# todo
def evaluation(paras, model, idx_to_label: dict):

    model.eval()

    test_dataset = SeqLabeling_Dataset(paras.test_data, paras.label_file)
    label_to_index = test_dataset.label_to_index

    test_dataloader = DataLoader(test_dataset, batch_size=paras.batch_size,
                            shuffle=False, drop_last=paras.drop_last)





def train(paras):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = SeqLabeling_Dataset(paras.train_data, paras.label_file)
    label_to_index = train_dataset.label_to_index

    train_dataloader = DataLoader(train_dataset, batch_size=paras.batch_size,
                                  shuffle=paras.shuffle, drop_last=paras.drop_last)

    # load model
    print(f'Loading model: {paras.model_name}.')
    tokenizer = BertTokenizerFast.from_pretrained(paras.model_name)
    model = BERT_CRF(paras)

    if paras.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=paras.learning_rate, momentum=paras.momentum)
    elif paras.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=paras.learning_rate, weight_decay=paras.weight_decay)
    else:
        raise ValueError('optimizer not supported.')

    # start to train.
    for idx, batch in enumerate(train_dataloader):

        batch_data, batch_label = batch

        batch_data_list = [data.split('&&&') for data in batch_data]
        batch_label_list = [label.split('&&&') for label in batch_label]

        encoded_input = tokenizer(batch_data_list,
                                  return_offsets_mapping=True,
                                  max_length=paras.max_length,
                                  truncation=True,
                                  is_split_into_words=True,
                                  padding=True,
                                  return_tensors='pt').to(device)

        # test
        # input_ids = encoded_input['input_ids']
        # token_type_ids = encoded_input['token_type_ids']
        # attention_mask = encoded_input['attention_mask']
        # offset_mapping = encoded_input['offset_mapping']

        batch_offset_mapping = encoded_input['offset_mapping']

        batch_adjust_label_list = batch_adjust_label(batch_offset_mapping,
                                                     batch_label_list)

        # for i in range(len(batch_offset_mapping)):
        #     # print(len(batch_offset_mapping[i]), len(batch_adjust_label_list[i]), len(input_ids[i]))
        #     for j in range(len(batch_offset_mapping[ i ])):
        #         print(f'{tokenizer.decode(input_ids[ i ][ j ])}, {batch_adjust_label_list[ i ][ j ]}')

        batch_label_idx = []
        for label_list in batch_adjust_label_list:
            batch_label_idx.append([label_to_index[label] for label in label_list])

        batch_input_ids = torch.LongTensor(encoded_input['input_ids']).to(device)
        batch_label_idx = torch.LongTensor(batch_label_idx).to(device)
        batch_attention_mask = torch.LongTensor(encoded_input['attention_mask']).to(device)

        # todo: acc, example
        loss = model(encoded_input, batch_label_idx)

        print(f'{loss.item():.3f}')
        # todo: gradient accumulation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # todo: loss log.
        # todo: evaluation


if __name__ == '__main__':

    paras = args()


    train(paras)




