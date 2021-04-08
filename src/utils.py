# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 31/03/2021 9:37
@Author: XINZHI YAO
"""

import os
import torch
import random
import logging
import numpy as np


# NOTSET DEBUG INFO WARNING ERROR CRITICAL FATAL
logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)

def batch_adjust_label(batch_offset_mapping: list, batch_label_list: list):
    # Batch adjust label by offset information from BertTokenizerFast
    batch_adjust_label_list = []

    for offset_mapping, label_list in zip(batch_offset_mapping, batch_label_list):
        adjust_label_list = adjust_label_by_offset(offset_mapping, label_list)
        batch_adjust_label_list.append(adjust_label_list)
    return batch_adjust_label_list


def adjust_label_by_offset(offset_mapping: list, label_list: list):
    # Adjust label by offset information from BertTokenizerFast
    adjust_label_list = [] # [CLS]
    label_idx = 0
    last_label = label_list[label_idx]
    label_list.insert(0, 'O')
    for idx, offset in enumerate(offset_mapping):

        if label_idx >= len(label_list):
            adjust_label_list.append('O')
            continue

        if offset[0] == 0:
            current_label = label_list[label_idx]
            adjust_label_list.append(current_label)
            label_idx += 1
            last_label = current_label
            continue

        if last_label.startswith('B-'):
            current_label = last_label.replace('B', 'I')
            adjust_label_list.append(current_label)
            last_label = current_label
        elif last_label.startswith('I-') or last_label == 'O':
            adjust_label_list.append(last_label)
        else:
            raise TypeError(f'Unknown label: {label_list[label_idx]}')

    return adjust_label_list

def tensor_to_list(tensor):
    # convert tensor to tensor.
    return tensor.numpy().tolist()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def boolean_string(s):
    # false or true in argparse
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def label_padding(batch_max_length: int, batch_label_list: list,
                  label_to_index:dict):

    out_idx = label_to_index['O']
    batch_label_idx = convert_label_to_index(batch_label_list, label_to_index)

    batch_label_pad = []
    for label_list in batch_label_idx:
        if len(label_list) > batch_max_length:
            batch_label_pad.append(label_list[:batch_max_length])
        else:
            pad_list = label_list + ([out_idx] * (batch_max_length - len(label_list)))
            batch_label_pad.append(pad_list)
    return batch_label_pad


def convert_label_to_index(batch_label_list: list, label_to_index: dict):
    batch_label_idx = []
    for label_list in batch_label_list:
        batch_label_idx.append([label_to_index[label] for label in label_list])
    return batch_label_idx


def convert_index_to_label(batch_index_list: list, index_to_label: dict,
                           del_special_token=True):
    batch_label_list = []
    for idx_list in batch_index_list:
        label_list = [index_to_label[idx] for idx in idx_list]
        if del_special_token:
            label_list = label_list[1:-1]
        batch_label_list.append(label_list)
    return batch_label_list


def convert_index_to_token(batch_index_list: list, tokenizer):
    batch_token_list = []
    for idx_list in batch_index_list:
        batch_token_list.append([''.join(tokenizer.decode(idx).split()) for idx in idx_list])
    return batch_token_list


def label_truncation(batch_label_list: list, max_length: int):
    process_label_list = []
    for label_list in batch_label_list:
        if len(label_list) > max_length:
            process_label_list.append(label_list[:max_length-2])
        else:
            process_label_list.append(label_list)
    return process_label_list


def batch_data_processing(batch_data_list: list, max_length: int, pad_idx: int, cls_idx: int, sep_idx: int
                        ,return_tensor=True, ):
    batch_length = [len(token_list) for token_list in batch_data_list]
    batch_max_length = max(batch_length)

    batch_size = len(batch_data_list)

    if batch_max_length > max_length:
        max_length = max_length
    else:
        max_length = batch_max_length

    mask = torch.ByteTensor(batch_size, max_length).fill_(0)
    for i in range(batch_size):
        mask[i, :batch_length[i]] = 1

    batch_data_tensor = [ ]
    for batch_data in batch_data_list:
        if len(batch_data) > (max_length - 2):
            batch_data = list(map(int, batch_data))[ :max_length - 2 ]
            batch_data.insert(0, cls_idx)
            batch_data.append(sep_idx)
            batch_data_tensor.append(batch_data)
        else:
            batch_data = list(map(int, batch_data))
            batch_data.insert(0, cls_idx)
            batch_data.append(sep_idx)
            batch_data = batch_data + ([ pad_idx ] * (max_length - len(batch_data)))
            batch_data_tensor.append(batch_data)

    if return_tensor:
        batch_data_tensor = torch.LongTensor(batch_data_tensor)

    return batch_data_tensor, mask
