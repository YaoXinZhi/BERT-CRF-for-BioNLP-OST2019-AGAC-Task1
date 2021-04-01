# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 31/03/2021 9:38
@Author: XINZHI YAO
"""


class args:
    def __init__(self):
        # self.data_path = 'H:/AGAC_NER/data/test_data.txt'
        self.train_data = '../data/train_input.txt'
        self.test_data = '../data/test_input.txt'

        self.label_file = '../data/label.txt'

        # fixme: biobert-base-cased
        self.model_name = 'bert-base-cased'

        self.batch_size = 5
        self.shuffle = True
        self.drop_last = False
        self.max_length = 128

        self.hidden_size = 768
        self.num_tags = 25

        self.learning_rate = 0.05
        self.optimizer = 'adam'
        self.weight_decay = 0.01
        self.momentum = 0.05

        self.droupout_prob = 0.3



