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
