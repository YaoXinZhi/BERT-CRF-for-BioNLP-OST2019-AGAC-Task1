# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 30/03/2021 16:19
@Author: XINZHI YAO
"""

import torch
from TorchCRF import CRF
from transformers import BertModel, BertTokenizer, BertPreTrainedModel
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, RandomSampler

import argparse

class SeqLabeling_Dataset(Dataset):
    def __init__(self, data_path):

        self.data_path = data_path
        self.data = []

        self.read_data()
        print(f'Data Path: {self.data_path},'
              f'Data size: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        return self.data[item]

    def read_data(self):
        data_list = []
        label_list = []

        with open(self.data_path) as f:
            for line in f:
                # fixme: split
                l = line.strip().split()
                if len(l) < 1:
                    if data_list and label_list:
                        self.data.append((data_list, label_list))
                        data_list = []
                        label_list = []
                else:
                    print(l)
                    token, label = l
                    data_list.append(token)
                    label_list.append(label)
            if (data_list, label_list) not in self.data:
                self.data.append((data_list, label_list))

def tensor_to_list(tensor):
    return tensor.numpy().tolist()

class config:
    def __init__(self):
        self.data_path = 'H:/AGAC_NER/data/test_data.txt'
        self.batch_size = 2
        self.hidden_size = 768
        self.num_tags = 12
        self.model_name = 'bert-base-cased'
        self.droupout_prob = 0.3


class BERT_CRF(nn.Module):

    def __init__(self, config):
        super(BERT_CRF, self).__init__()

        self.num_tags = config.num_tags
        self.hidden_size = config.hidden_size

        self.bert_layer = BertModel.from_pretrained(args.model_name)
        self.dropout = nn.Dropout(args.droupout_prob)

        self.hidden_to_tag_layer = nn.Linear(args.hidden_size, args.num_tags)

        self.crf_layer = CRF(args.num_tags)

    def tag_outputs(self, encoded_input):

        batch_input_ids = torch.LongTensor(encoded_input[ 'input_ids' ])
        batch_token_type_ids = torch.LongTensor(encoded_input[ 'token_type_ids' ])
        batch_attention_mask = torch.LongTensor(encoded_input[ 'attention_mask' ])

        # outputs = self.bert_layer(**encoded_input)
        outputs = self.bert_layer(batch_input_ids,
                                  batch_token_type_ids,
                                  batch_attention_mask)

        last_layer_output = outputs[0]

        last_layer_output = self.dropout(last_layer_output)

        # batch_size, seq_len, tag_num
        emissions = self.hidden_to_tag_layer(last_layer_output)
        return emissions

    def forward(self, encoded_input, labels):

        attention_mask = torch.LongTensor(encoded_input['attention_mask'])

        emissions = self.tag_outputs(encoded_input)

        # todo: check the algorithm
        # loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte())
        loss = self.crf(emissions, labels, mask=attention_mask.byte())

        return loss

    def predict(self, encoded_input):

        attention_mask = encoded_input['attention_mask']

        emissions = self.tag_outputs(encoded_input)
        return self.crf.viterbi_decode(emissions, attention_mask.byte())

if __name__ == '__main__':


    # parser = argparse.ArgumentParser(description='BERT+CRF for NER.')
    #
    # args = parser.parse_args()

    args = config()


    NER_dataset = SeqLabeling_Dataset(args.data_path)

    concept_dataloader = DataLoader(dataset=NER_dataset,
                                    batch_size=args.batch_size,
                                    drop_last=False,
                                    shuffle=False)

    for idx, batch_data in enumerate(concept_dataloader):

        data, labels = batch_data
        print(f'{idx}')
        print(data)
        print(labels)

    label_to_idx = {
        'O': 0,
        'B-LOC': 1,
        'I-LOC': 2,
    }

    idx_to_label = {v:k for k, v in label_to_idx.items()}

    data_list = ['Its','headquarters','are','in','DUMBO',',','therefore','very','close',
                    'to','the','Manhattan','Bridge','.']
    label_idx_list = [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 0 ]

    label_list = [idx_to_label[idx] for idx in label_idx_list]

    label_idx_list = torch.LongTensor(label_idx_list)

    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    # fixme: add [cls]/[sep] label
    encoded_input = tokenizer(data_list, is_split_into_words=True,
                              return_tensors='pt', padding=True, )

    tokenizer.decode(encoded_input[ "input_ids" ].numpy().tolist()[0])

    model = BERT_CRF(args)
