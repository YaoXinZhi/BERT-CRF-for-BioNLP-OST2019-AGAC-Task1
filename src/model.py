# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 31/03/2021 9:33
@Author: XINZHI YAO
"""

import torch
import torch.nn as nn
from TorchCRF import CRF
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import BertModel, BertTokenizer, BertPreTrainedModel


class BERT_CRF(nn.Module):

    def __init__(self, paras):
        super(BERT_CRF, self).__init__()

        self.num_tags = paras.num_tags
        self.hidden_size = paras.hidden_size

        self.bert_layer = BertModel.from_pretrained(paras.model_name)
        self.dropout = nn.Dropout(paras.droupout_prob)

        self.hidden_to_tag_layer = nn.Linear(paras.hidden_size, paras.num_tags)

        self.crf_layer = CRF(paras.num_tags, batch_first=True)

    def tag_outputs(self, batch_encoded_inputs):

        # outputs = self.bert_layer(**encoded_input)
        batch_input_ids = batch_encoded_inputs['input_ids']
        batch_type_ids = batch_encoded_inputs['token_type_ids']
        batch_attention_mask = batch_encoded_inputs['attention_mask']

        outputs = self.bert_layer(input_ids = batch_input_ids,
                                  attention_mask=batch_attention_mask,
                                  token_type_ids=batch_type_ids)

        # batch_size, seq_length, hidden_dim
        last_layer_output = outputs[0]

        last_layer_output = self.dropout(last_layer_output)

        # batch_size, seq_length, num_labels
        emissions = self.hidden_to_tag_layer(last_layer_output)
        return emissions

    def forward(self, batch_encoded_inputs, batch_label_idx):
        """
        emissions: batch_size, seq_length, num_labels
        batch_label_idx: batch_size, seq_length
        batch_attention_mask: batch_size, seq_length
        """
        batch_attention_mask = batch_encoded_inputs['attention_mask']
        emissions = self.tag_outputs(batch_encoded_inputs)

        loss = self.crf_layer.forward(emissions, batch_label_idx, batch_attention_mask.byte())

        return loss

    def predict(self, encoded_input):

        attention_mask = encoded_input['attention_mask']

        emissions = self.tag_outputs(encoded_input)
        return self.crf.viterbi_decode(emissions, attention_mask.byte())

