# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 31/03/2021 9:38
@Author: XINZHI YAO
"""

import logging

import torch.cuda
from transformers import WEIGHTS_NAME, CONFIG_NAME

class args:
    def __init__(self):

        # global parameters
        self.mode = 'train'
        self.direct_infer = True
        self.load_model_from_checkpoint = False
        self.use_cpu = False
        self.device = 'cuda' if torch.cuda.is_available() and (not self.use_cpu) else 'cpu'

        # data parameters
        # self.train_data = '../data/train_input.txt'
        self.train_data = '../data/train.agac.v3.txt'
        self.test_data = '../data/test_input.txt'
        self.infer_data = '../data/pubtator_file/ad.bio.txt'
        self.label_file = '../data/label.txt'
        #self.label_file = '../data/label_non_pubtator_label.txt'
        # self.label_file = '../data/label.only-reg.txt'
        self.load_labeled_data = True
        self.only_loaded_label = True
        self.num_tags = 26
        self.use_word_piece = True
        # self.use_word_piece = False
        self.special_token_label = 'O'

        # model load parameters
        self.seed = 126

        self.model_name = 'dmis-lab/biobert-base-cased-v1.1'
        #self.model_name = 'bert-base-cased'
        self.do_lower_case = False
        self.use_fast = True

        self.unk_token = '[UNK]'
        self.seq_token = '[SEP]'
        self.cls_token = '[CLS]'

        # dataloader parameters
        self.batch_size = 4
        self.shuffle = True
        self.drop_last = False
        self.max_length = 512

        # model parameters
        self.hidden_size = 768
        #self.hidden_size = 1024
        # fixme: 0.3
        self.dropout_prob = 0.3
        # self.num_tags = 25
        # fixme: add [PAD]
        #self.num_tags = 16

        # training parameters
        self.learning_rate = 5e-5
        # sgd adam adamw
        self.optimizer = 'adamw'
        self.weight_decay = 1e-5
        self.num_train_epochs = 100
        self.warmup_steps = 10
        self.adam_epsilon = 1e-8
        self.num_warmup_steps=50

        # inference parameters
        self.eval_fine_tuning_model = True
        self.infer_save_file = '../infer_result/ad.word-piece.infer_output.txt'

        # logging parameters
        self.print_example = False
        self.save_log_file = True
        # NOTSET, DEBUG, INFO,
        self.logging_level = logging.INFO
        self.log_save_path = '../logging'
        self.model_save_path = '../models'
        self.model_save_name = WEIGHTS_NAME
        self.config_save_name = CONFIG_NAME
        self.log_file = 'NER.word-piece.log'




