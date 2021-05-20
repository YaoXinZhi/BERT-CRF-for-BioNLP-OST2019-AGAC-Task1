# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 31/03/2021 9:33
@Author: XINZHI YAO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from TorchCRF import CRF
from torch.utils.data import Dataset, DataLoader, RandomSampler

from transformers import BertModel, BertTokenizer, BertPreTrainedModel


# class BertCRFTagger(nn.Module):
class BertCRFTagger(BertPreTrainedModel):

    def __init__(self, bert_config, bert, hidden_size, num_tags, dropout):
        # super().__init__()
        super(BertCRFTagger, self).__init__(bert_config)
        self.bert = bert
        try:
            self.crf = CRF(num_tags, batch_first=True)
        except:
            self.crf = CRF(num_tags)
        self.fc = nn.Linear(hidden_size, num_tags)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, mask, tags=None):
        bert_output = self.bert(input_ids)
        last_hidden_state = bert_output['hidden_states'][ -1 ]

        emission = self.fc(last_hidden_state)

        if tags is not None:
            # loss = -self.crf(torch.log_softmax(emission, dim=2), tags, mask=mask, reduction='mean')
            loss = -self.crf(torch.log_softmax(emission, dim=2), tags, mask=mask).mean()
            return loss
        else:
            try:
                prediction = self.crf.decode(emission, mask=mask)
            except:
                prediction = self.crf.viterbi_decode(emission, mask=mask)

            return prediction
