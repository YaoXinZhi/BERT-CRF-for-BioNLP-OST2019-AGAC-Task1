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
from nltk import tokenize
from transformers import BertTokenizerFast

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

def label_padding_with_special_token(batch_length: int, batch_label_list: list,
                                     label_to_index: dict, return_tensor=True):

    pad_idx = label_to_index['[PAD]']
    batch_label_idx = convert_label_to_index(batch_label_list, label_to_index)

    batch_label_pad = []
    for label_list in batch_label_idx:
        if len(label_list) > batch_length:
            # fixme: error
            print(batch_length)
            print(len(label_list))
            label_list = label_list[:batch_length]
            batch_label_pad.append(label_list)
        else:
            label_list = label_list + ([pad_idx] * (batch_length-len(label_list)))
            batch_label_pad.append(label_list)

    if return_tensor:
        return torch.LongTensor(batch_label_pad)
    return batch_label_pad



def label_padding(seq_max_length: int, batch_max_length: int, batch_label_list: list,
                  label_to_index: dict, return_tensor=True,
                  special_token_label:str= '[PAD]'):

    if batch_max_length > seq_max_length:
        max_length = seq_max_length
    else:
        max_length = batch_max_length

    # fixme: 'O' or '[PAD]'
    pad_idx = label_to_index[special_token_label ]
    batch_label_idx = convert_label_to_index(batch_label_list, label_to_index)

    batch_label_pad = [ ]
    for label_list in batch_label_idx:
        if len(label_list) > (max_length - 2):
            label_list = label_list[ :max_length - 2 ]
            label_list.insert(0, pad_idx)
            label_list.append(pad_idx)
            batch_label_pad.append(label_list)
        else:
            label_list.insert(0, pad_idx)
            label_list.append(pad_idx)
            label_list = label_list + ([ pad_idx ] * (max_length - len(label_list)))
            batch_label_pad.append(label_list)

    if return_tensor:
        return torch.LongTensor(batch_label_pad)

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


def convert_index_to_token(batch_index_list: list, tokenizer, mask:list=None,
                           del_special_token:bool=False):
    batch_token_list = []
    if not mask is None:
        if len(batch_index_list) != len(mask):
            raise TypeError('len(batch_index_list) != len(mask)')
    for idx, idx_list in enumerate(batch_index_list):
        if not mask is None:
            token_len = sum(mask[idx])
            token_list = [ ''.join(tokenizer.decode(int(idx)).split()) for idx in idx_list[:token_len]]
        else:
            token_list = [ ''.join(tokenizer.decode(int(idx)).split()) for idx in idx_list]

        if del_special_token:
            token_list = token_list[1:-1]
        batch_token_list.append(token_list)
    return batch_token_list


def convert_index_to_label_single(index_list: list, index_to_label: dict):
    return [index_to_label[idx] for idx in index_list]

def convert_index_to_token_single(index_list: list, tokenizer):
    return [tokenizer.decode(index) for index in index_list]


def label_truncation(batch_label_list: list, max_length: int, del_special_token:bool=False):
    process_label_list = []
    for label_list in batch_label_list:
        if len(label_list) >= max_length-2:
            label_list = label_list[:max_length-2]
            if del_special_token:
                label_list = label_list[1:-1]
            process_label_list.append(label_list)
        else:
            # fixme: 0510
            if del_special_token:
                label_list = label_list[1: -1]
            process_label_list.append(label_list)
            # process_label_list.append(label_list[:-2])
    return process_label_list


def batch_data_processing(batch_data_list: list, max_length: int, pad_idx: int, cls_idx: int, sep_idx: int
                        ,return_tensor=True, ):
    # fixme: 20210510 batch_max_length + 2 [CLS] [SEP]
    batch_length = [len(token_list) + 2 for token_list in batch_data_list]
    batch_max_length = max(batch_length)

    batch_size = len(batch_data_list)

    max_length = min(batch_max_length, max_length)

    mask = torch.ByteTensor(batch_size, max_length).fill_(0)
    for i in range(batch_size):
        mask[i, :batch_length[i]] = 1

    batch_data_tensor = [ ]
    for batch_data in batch_data_list:
        if len(batch_data) > (max_length - 2):
            batch_data = list(map(int, batch_data))[:max_length - 2 ]
            batch_data.insert(0, cls_idx)
            batch_data.append(sep_idx)
            batch_data_tensor.append(batch_data)
        else:
            batch_data = list(map(int, batch_data))
            batch_data.insert(0, cls_idx)
            batch_data.append(sep_idx)
            batch_data = batch_data + ([pad_idx] * (max_length - len(batch_data)))
            batch_data_tensor.append(batch_data)

    if return_tensor:
        batch_data_tensor = torch.LongTensor(batch_data_tensor)

    return batch_data_tensor, mask

def batch_data_wordpiece_processing(tokenizer: BertTokenizerFast, batch_data_list: list,
                                    max_length: int, batch_label_list: list,
                                    special_token_label_idx: str='[PAD]',
                                    inference:bool=False):

    encoded = tokenizer(batch_data_list,
                       return_offsets_mapping=True,
                       max_length=max_length,
                       truncation=True,
                       padding=True,
                       is_split_into_words=True,
                        return_tensors='pt')

    batch_input_ids = encoded['input_ids']
    batch_attention_mask = encoded['attention_mask']
    batch_offset_mapping = encoded['offset_mapping']
    # if inference:
    #     return batch_input_ids, batch_attention_mask.byte()


    batch_adjust_label = []
    for attention_mask, label_list, offset_mapping in zip(batch_attention_mask, batch_label_list, batch_offset_mapping):
    # function
        adjust_label_list = []
        label_idx = 0
        token_length = sum(attention_mask)-1
        last_label = label_list[label_idx]
        # [CLS] label
        if not inference:
            adjust_label_list.append(special_token_label_idx)
        for offset in offset_mapping[1:token_length]:
            #print(offset)
            if offset[0] == 0:
                #print(label_idx, label_list[label_idx])
                current_label = label_list[label_idx]
                adjust_label_list.append(current_label)
                label_idx += 1
                last_label = current_label
                continue

            if inference:
                current_label = last_label
                adjust_label_list.append(current_label)
                continue

            # offset[0] != 0
            if last_label.startswith('B-'):
                current_label = last_label.replace('B', 'I')
                adjust_label_list.append(current_label)
                last_label = current_label
            elif last_label.startswith('I-') or last_label == 'O':
                current_label = last_label
                adjust_label_list.append(current_label)

        # [SEP] label
        if not inference:
            adjust_label_list.append(special_token_label_idx)

        batch_adjust_label.append(adjust_label_list)

    return batch_input_ids, batch_attention_mask.byte(), batch_adjust_label



def batch_data_truncate(batch_token_list: list, max_length: int):
    process_token_list = []
    for token_list in batch_token_list:
        if len(token_list) >= max_length-2:
            process_token_list.append(token_list[:max_length-2])
        else:
            process_token_list.append(token_list)

    return process_token_list

def get_sent_offset(doc: str):
    sent_list = tokenize.sent_tokenize(doc)
    sent_to_offset = {}
    sent_to_id = {}
    for sent in sent_list:
        begin = doc.find(sent)
        end = begin + len(sent)
        sent_to_offset[sent] = (begin, end)
        sent_to_id[sent] = len(sent_to_id)

    for sent, (begin, end) in sent_to_offset.items():
        if doc[begin: end]!= sent:
            print('Warning: Position calculation error.')
            print(doc)
            print(doc[begin: end], sent)
    return sent_to_offset, sent_to_id



def get_token_offset(sentence: str):
    token_to_offset = {}

    start = 0
    for token in word_tokenize(sentence):

        token_start = sentence.find(token, start)
        token_end = token_start + len(token)
        token_to_offset[token] = (token_start, token_end)
    return token_to_offset

def show_example(input_ids: list, batch_label: list, tokenizer):
    for ids, label_list in zip(input_ids, batch_label):
        decode_token = [ tokenizer.decode(_id) for _id in ids ]
        for token, label in zip(decode_token, label_list):
            print(token, label)
        print()
