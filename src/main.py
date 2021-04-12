# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 07/04/2021 12:53
@Author: XINZHI YAO
"""

import os
import time
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import BertTokenizerFast, BertModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from TorchCRF import CRF

from model import BertCRFTagger
from utils import *
from dataloader import SeqLabeling_Dataset
from config import args
from conll_eval import evaluate

def evaluation(model, data_loader, index_to_label, vocab_dict, paras, device):
    """
    Contributor:
        QianQian Peng: conlleval.pl evaluation.
        Sizhuo Oyang: Conllevla.pl evaluation.
    """
    model.eval()

    total_pred_label = []
    total_ture_label = []
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            batch_data, batch_label = batch
            batch_data_list = [data.split('&&&') for data in batch_data]
            batch_label_list = [label.split('&&&') for label in batch_label]

            input_ids, mask = batch_data_processing(batch_data_list, paras.max_length,
                                                            vocab_dict.get('[PAD]'),
                                                            vocab_dict.get('[CLS]'),
                                                            vocab_dict.get('[SEP]'))

            input_ids = input_ids.to(device)
            mask = mask.to(device)

            batch_max_length = input_ids.shape[1]

            predict_result = model(input_ids, mask)

            predict_label_list = convert_index_to_label(predict_result, index_to_label)
            ture_label_list = label_truncation(batch_label_list, batch_max_length)

            if args.print_example:
                logger.debug('Example:')
                logger.debug(f'predict: {predict_label_list[0]}')
                logger.debug(f'ture: {ture_label_list[0]}')

            for predict_list, ture_list in zip(predict_label_list, ture_label_list):
                if len(predict_list) != len(ture_list):
                    logger.debug('different length.')
                    logger.debug(f'predict: {len(predict_list)}, ture: {len(ture_list)}')
                    logger.debug(f'{predict_list}\n{ture_list}')
                    continue
                total_pred_label.extend(predict_list)
                total_ture_label.extend(ture_list)

    logger.debug(f'total ture_label: {len(total_ture_label)}, '
                 f'total pred_label: {len(total_pred_label)}')

    (precision, recall, f_score), acc_non_o, acc_inc_o,  = evaluate(total_ture_label, total_pred_label)
    return acc_non_o, acc_inc_o, precision, recall, f_score


def main(paras):

    logger = logging.getLogger(__name__)
    if args.save_log_file:
        logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = paras.logging_level,
                            filename=f'{paras.log_save_path}/{paras.log_file}',
                            filemode='w')
    else:
        logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = paras.logging_level,)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f'Loading model: {paras.model_name}.')
    tokenizer = BertTokenizerFast.from_pretrained(paras.model_name)
    bert = BertModel.from_pretrained(paras.model_name, output_hidden_states=True)

    vocab_dict = tokenizer.get_vocab()

    train_dataset = SeqLabeling_Dataset(paras.train_data, paras.label_file, vocab_dict)
    label_to_index = train_dataset.label_to_index
    index_to_label = train_dataset.index_to_label

    train_dataloader = DataLoader(train_dataset, batch_size=paras.batch_size,
                                  shuffle=paras.shuffle, drop_last=paras.drop_last)

    test_dataset = SeqLabeling_Dataset(paras.test_data, paras.label_file, vocab_dict)
    test_dataloader = DataLoader(test_dataset, batch_size=paras.batch_size,
                                 shuffle=paras.shuffle, drop_last=paras.drop_last)

    bert_crf_tagger = BertCRFTagger(bert, paras.hidden_size, paras.num_tags,
                                paras.droupout_prob).to(device)

    if paras.optimizer == 'adam':
        logger.info('Loading Adam optimizer.')
        optimizer = torch.optim.Adam(bert_crf_tagger.parameters(), lr=paras.learning_rate)
    elif paras.optimizer == 'adamw':
        logger.info('Loading AdamW optimizer.')
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [ p for n, p in bert_crf_tagger.named_parameters() if not any(nd in n for nd in no_decay) ],
              'weight_decay': 0.01},
            {'params': [ p for n, p in bert_crf_tagger.named_parameters() if any(nd in n for nd in no_decay) ],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=paras.learning_rate,
                          eps=args.adam_epsilon)
    else:
        logger.warning(f'optimizer must be "Adam" or "AdamW", but got {paras.optimizer}.')
        logger.info('Loading Adam optimizer.')
        optimizer = torch.optim.Adam(bert_crf_tagger.parameters(), lr=paras.learning_rate)

    best_eval = {'acc_non_O': 0 ,'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'loss': 0}
    for epoch in range(paras.num_train_epochs):
        epoch_loss = 0
        bert_crf_tagger.train()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            batch_data, batch_label = batch
            batch_data_list = [ data.split('&&&') for data in batch_data ]
            batch_label_list = [ label.split('&&&') for label in batch_label ]

            input_ids, mask = batch_data_processing(batch_data_list, paras.max_length,
                                                            vocab_dict.get('[PAD]'),
                                                            vocab_dict.get('[CLS]'),
                                                            vocab_dict.get('[SEP]'))
            input_ids = input_ids.to(device)
            mask = mask.to(device)

            batch_max_length = input_ids.shape[1]
            batch_label_pad = label_padding(paras.max_length,batch_max_length, batch_label_list,
                                            label_to_index)

            batch_label_pad = torch.LongTensor(batch_label_pad)

            loss = bert_crf_tagger(input_ids, mask, batch_label_pad)

            epoch_loss += loss.detach().cpu().item()

            logger.info(f'epoch: {epoch}, step: {step}, loss: {loss:.4f}')

            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / len(train_dataloader)

        acc_non_O, acc, precision, recall, f1 = evaluation(bert_crf_tagger, test_dataloader,
                                                index_to_label, vocab_dict, paras, device)

        logger.info(f'Epoch: {epoch}, Epoch-Average Loss: {epoch_loss}')
        logger.info(f'ACC_non_O: {acc_non_O:.4f}, ACC_inc_O: {acc:.4f}, Precision: {precision:.4f}, '
              f'Recall: {recall:.4f}, F1-score: {f1:.4f}')

        if best_eval['loss'] == 0 or f1 > best_eval['f1']:
            best_eval['loss'] = epoch_loss
            best_eval['acc'] = acc
            best_eval['acc_non_O'] = acc_non_O
            best_eval['precision'] = precision
            best_eval['recall'] = recall
            best_eval['f1'] = f1
            torch.save(bert_crf_tagger, f'{paras.log_save_path}/{paras.model_save_name}')

            with open(f'{paras.log_save_path}/checkpoint.log', 'w') as wf:
                wf.write(f'Save time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')
                wf.write(f'Best F1-score: {best_eval["f1"]:.4f}\n')
                wf.write(f'Precision: {best_eval["precision"]:.4f}\n')
                wf.write(f'Recall: {best_eval["recall"]:.4f}\n')
                wf.write(f'Accuracy(include-O): {best_eval["acc"]:.4f}\n')
                wf.write(f'Accuracy(none-O): {best_eval["acc_non_O"]:.4f}\n')
                wf.write(f'Epoch-Average Loss: {best_eval["loss"]:.4f}\n')

            logger.info(f'Updated model, best F1-score: {best_eval["f1"]:.4f}\n')

    logger.info(f'Train complete, Best F1-score: {best_eval["f1"]:.4f}.')


if __name__ == '__main__':

    args = args()

    set_seed(args.seed)

    main(args)
