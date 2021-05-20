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

from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import BertTokenizerFast, BertModel, BertTokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers import BertConfig
from transformers import AdamW #, get_linear_schedule_with_warmup

from TorchCRF import CRF

from model import BertCRFTagger
from utils import *
from dataloader import SeqLabeling_Dataset, Infer_Dataset
from config import args
from conll_eval import evaluate


# todo: add tokenizer option for debug.
def evaluation(model, data_loader, index_to_label, vocab_dict,
               paras, device, tokenizer: BertTokenizer):
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

            if paras.use_word_piece:
                # fixme: [PAD] or O or special token
                input_ids, mask, ture_label_list = batch_data_wordpiece_processing(tokenizer, batch_data_list,
                                                                                      paras.max_length,
                                                                                      batch_label_list,
                                                                                      paras.special_token_label)
            else:
                input_ids, mask = batch_data_processing(batch_data_list, paras.max_length,
                                                                vocab_dict.get('[PAD]'),
                                                                vocab_dict.get('[CLS]'),
                                                                vocab_dict.get('[SEP]'))
                batch_max_length = input_ids.shape[1]
                ture_label_list = label_truncation(batch_label_list, batch_max_length)

            input_ids = input_ids.to(device)
            mask = mask.to(device)

            predict_result = model(input_ids, mask)

            if paras.use_word_piece:
                predict_label_list = convert_index_to_label(predict_result, index_to_label,
                                                            del_special_token=False)
            else:
                predict_label_list = convert_index_to_label(predict_result, index_to_label,
                                                            del_special_token=True)

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


def inference(paras):

    # load model from fine-tuning
    model, tokenizer = _load_fine_tuning_model(paras.model_save_path, paras.model_save_name,
                                               paras.use_fast)

    vocab_dict = tokenizer.get_vocab()

    model.eval()
    model.to(paras.device)
    infer_dataset = Infer_Dataset(paras.infer_data, paras.label_file, vocab_dict)
    index_to_label = infer_dataset.index_to_label
    infer_dataloader = DataLoader(infer_dataset, batch_size=paras.batch_size,
                                  shuffle=False, drop_last=paras.drop_last)

    if paras.eval_fine_tuning_model:
        acc_non_O, acc, precision, recall, f1 = evaluation(bert_crf_tagger, test_dataloader,
                                                           index_to_label, vocab_dict,
                                                           paras, device, tokenizer)
        logger.info('Evaluation of fine-tuning model:')
        logger.info(f'ACC_non_O: {acc_non_O:.4f}, ACC_inc_O: {acc:.4f}, Precision: {precision:.4f}, '
                    f'Recall: {recall:.4f}, F1-score: {f1:.4f}')

    wf = open(paras.infer_file, 'w')
    with torch.no_grad():
        for step, batch in enumerate(infer_dataloader):
            batch_data, batch_orig_data = batch
            batch_data_list = [data.split('&&&') for data in batch_data]
            batch_orig_data = [data.split('&&&') for data in batch_orig_data]

            input_ids, mask = batch_data_processing(batch_data_list, paras.max_length,
                                                    vocab_dict.get('[PAD]'),
                                                    vocab_dict.get('[CLS]'),
                                                    vocab_dict.get('[SEP]'))
            input_ids = input_ids.to(paras.device)
            mask = mask.to(paras.device)

            predict_result = model(input_ids, mask)

            predict_label_list = convert_index_to_label(predict_result, index_to_label)

            if len(batch_orig_data) != len(predict_label_list):
                raise ValueError('Difference length between batch_orig_data and predict_label_list.')

            for token_list, label_list in zip(batch_orig_data, predict_label_list):
                if len(token_list) != len(label_list):
                    logger.debug('different length.')
                    logger.debug(f'tokens: {len(token_list)}, labels: {len(label_list)}')
                    logger.debug(f'{token_list}\n{label_list}')
                    continue
                for token, label in zip(token_list, label_list):
                    wf.write(f'{token}\t{label}\n')
                wf.write('\n')

    wf.close()
    logger.info(f'Inference file: {paras.infer_file} save done.')



def _direct_inference(model: BertModel, tokenizer: BertTokenizer, infer_dataloader: DataLoader,
                      index_to_label: dict, vocab_dict: dict, paras):

    wf = open(paras.infer_save_file, 'w', encoding='utf-8')
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(infer_dataloader):
            batch_data, batch_orig_data = batch

            batch_data_list = [data.split('&&&') for data in batch_data
                               if data.split('&&&') != ['']]

            batch_orig_data_list = [data.split('&&&') for data in batch_orig_data
                                    if data.split('&&&') != ['']]

            if paras.use_word_piece:
                input_ids, mask, batch_orig_data_list = batch_data_wordpiece_processing(tokenizer,
                                                                                        batch_data_list,
                                                                                        paras.max_length,
                                                                                        batch_orig_data_list,
                                                                                        paras.special_token_label,
                                                                                        True)

                batch_token_list = convert_index_to_token(input_ids, tokenizer, mask, True)

            else:
                input_ids, mask = batch_data_processing(batch_data_list, paras.max_length,
                                                        vocab_dict.get('[PAD]'),
                                                        vocab_dict.get('[CLS]'),
                                                        vocab_dict.get('[SEP]'))
                batch_orig_data_list = batch_data_truncate(batch_orig_data_list, batch_max_length)

            input_ids = input_ids.to(paras.device)
            mask = mask.to(paras.device)

            batch_max_length = input_ids.shape[1]

            predict_result = model(input_ids, mask)

            predict_label_list = convert_index_to_label(predict_result, index_to_label,
                                                        del_special_token=True)

            if len(predict_label_list) != len(batch_orig_data_list):
                raise ValueError('Difference length between "predict_label_list" and "batch_orig_data_list".')

            # for label_list, orig_list in zip(predict_label_list, batch_orig_data_list):
            for orig_list, label_list in zip(batch_orig_data_list, predict_label_list):
                if len(orig_list) != len(label_list):
                    raise ValueError(f'Difference length between'
                                     f' "Label_list-{len(label_list)}" '
                                     f'and "orig_list-{len(orig_list)}".')

                if paras.use_word_piece:
                    for token_info, label in zip(orig_list, label_list):
                        token_wf = '\t'.join(token_info.split('$$$'))
                        wf.write(f'{token_wf}\t{label}\n')
                else:
                    for token_info, label in zip(orig_list, label_list):
                        token_wf = '\t'.join(token_info.split('$$$'))
                        wf.write(f'{token_wf}\t{label}\n')
                wf.write('\n')
    wf.close()


def _save_model(save_path: str, model: BertModel, tokenizer: BertTokenizer,
                model_save_name: str, config_save_name: str):

    model_save_path = os.path.join(save_path, model_save_name)
    config_save_path = os.path.join(save_path, config_save_name)

    model_to_save = model.module if hasattr(model, 'module') else model

    torch.save(model_to_save.state_dict(), model_save_path)
    model_to_save.config.to_json_file(config_save_path)
    tokenizer.save_vocabulary(save_path)

def _load_fine_tuning_model(model: BertModel, save_path: str, model_save_name: str, use_fast: bool = True):

    model_save_path = os.path.join(save_path, model_save_name)
    model.load_state_dict(torch.load(model_save_path))

    if use_fast:
        tokenizer = BertTokenizerFast.from_pretrained(save_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(save_path)
    return model, tokenizer

def _load_pre_trained_model(paras):
    if paras.use_fast:
        tokenizer = BertTokenizerFast.from_pretrained(paras.model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(paras.model_name)
    bert_config = BertConfig.from_pretrained(paras.model_name)
    bert = BertModel.from_pretrained(paras.model_name, output_hidden_states=True)
    bert_crf_tagger = BertCRFTagger(bert_config, bert, paras.hidden_size, paras.num_tags,
                                    paras.dropout_prob)
    return bert_crf_tagger, tokenizer

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

    device = 'cuda' if torch.cuda.is_available() and not paras.use_cpu else 'cpu'

    if paras.load_model_from_checkpoint:
        logger.info(f'Loading model from pretrained: {paras.model_save_path}.')
        bert_crf_tagger, _ = _load_pre_trained_model(paras)
        bert_crf_tagger, tokenizer = _load_fine_tuning_model(bert_crf_tagger, paras.model_save_path, paras.model_save_name, paras.use_fast)
    else:
        logger.info(f'Loading model: {paras.model_name}.')
        bert_crf_tagger, tokenizer = _load_pre_trained_model(paras)

    bert_crf_tagger.to(device)
    vocab_dict = tokenizer.get_vocab()

    train_dataset = SeqLabeling_Dataset(paras.train_data, paras.label_file, vocab_dict,
                                        only_loaded_label=paras.only_loaded_label,
                                        use_word_piece=paras.use_word_piece)
    train_dataloader = DataLoader(train_dataset, batch_size=paras.batch_size,
                                  shuffle=paras.shuffle, drop_last=paras.drop_last)
    label_to_index = train_dataset.label_to_index
    index_to_label = train_dataset.index_to_label


    test_dataset = SeqLabeling_Dataset(paras.test_data, paras.label_file,
                                       vocab_dict,
                                       only_loaded_label= paras.only_loaded_label,
                                       use_word_piece=paras.use_word_piece)
    test_dataloader = DataLoader(test_dataset, batch_size=paras.batch_size,
                                 shuffle=paras.shuffle, drop_last=paras.drop_last)

    # load infer data
    infer_dataloader = None
    if paras.direct_infer:
        logger.info('Loading Inference data.')
        infer_dataset = Infer_Dataset(paras.infer_data, paras.label_file,
                                      vocab_dict, use_word_piece=paras.use_word_piece)
        infer_dataloader = DataLoader(infer_dataset, batch_size=paras.batch_size,
                                      shuffle=False, drop_last=paras.drop_last)

    if paras.optimizer.lower() == 'adam':
        logger.info('Loading Adam optimizer.')
        optimizer = torch.optim.Adam(bert_crf_tagger.parameters(), lr=paras.learning_rate)
    elif paras.optimizer.lower() == 'adamw':
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

    logger.info('Training Start.')
    best_eval = {'acc_non_O': 0 ,'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'loss': 0}
    for epoch in range(paras.num_train_epochs):
        epoch_loss = 0
        bert_crf_tagger.train()
        for step, batch in enumerate(train_dataloader):
            # break
            optimizer.zero_grad()

            batch_data, batch_label = batch
            batch_data_list = [ data.split('&&&') for data in batch_data ]
            batch_label_list = [ label.split('&&&') for label in batch_label ]

            # todo: check
            if paras.use_word_piece:
                # fixme: [PAD] or O or special token
                input_ids, mask, batch_adjust_label = batch_data_wordpiece_processing(tokenizer, batch_data_list,
                                                                                      paras.max_length, batch_label_list,
                                                                                      paras.special_token_label)
                # test
                # show_example(input_ids, batch_adjust_label, tokenizer)

                batch_length = input_ids.shape[1]
                batch_label_pad = label_padding_with_special_token(batch_length, batch_adjust_label,
                                                                      label_to_index)


            else:

                input_ids, mask = batch_data_processing(batch_data_list, paras.max_length,
                                                                vocab_dict.get('[PAD]'),
                                                                vocab_dict.get('[CLS]'),
                                                                vocab_dict.get('[SEP]'))


                batch_max_length = input_ids.shape[1]
                batch_label_pad = label_padding(paras.max_length, batch_max_length, batch_label_list,
                                                label_to_index, special_token_label=paras.special_token_label)

            input_ids = input_ids.to(device)
            mask = mask.to(device)

            loss = bert_crf_tagger(input_ids, mask, batch_label_pad)

            epoch_loss += loss.detach().cpu().item()

            logger.info(f'epoch: {epoch}, step: {step}, loss: {loss:.4f}')

            loss.backward()
            optimizer.step()

            # fixme: delete debug evaluation
            # acc_non_O, acc, precision, recall, f1 = evaluation(bert_crf_tagger, test_dataloader,
            #                                                    index_to_label, vocab_dict, paras, device)
            # fixme: delete debug inference
            # if paras.direct_infer:
            #     logger.info('inference')
            #     _direct_inference(bert_crf_tagger, tokenizer, infer_dataloader,
            #                                   index_to_label, vocab_dict, paras)

        epoch_loss = epoch_loss / len(train_dataloader)
        logger.info('Evaluating.')
        acc_non_O, acc, precision, recall, f1 = evaluation(bert_crf_tagger, test_dataloader,
                                                index_to_label, vocab_dict, paras, device, tokenizer)

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

            with open(f'{paras.log_save_path}/checkpoint.log', 'w') as wf:
                wf.write(f'Save time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')
                wf.write(f'Best F1-score: {best_eval["f1"]:.4f}\n')
                wf.write(f'Precision: {best_eval["precision"]:.4f}\n')
                wf.write(f'Recall: {best_eval["recall"]:.4f}\n')
                wf.write(f'Accuracy(include-O): {best_eval["acc"]:.4f}\n')
                wf.write(f'Accuracy(none-O): {best_eval["acc_non_O"]:.4f}\n')
                wf.write(f'Epoch-Average Loss: {best_eval["loss"]:.4f}\n')

            logger.info(f'Updated model, best F1-score: {best_eval["f1"]:.4f}\n')
            _save_model(paras.model_save_path, bert_crf_tagger, tokenizer,
                        paras.model_save_name, paras.config_save_name)
            if paras.direct_infer and f1>50:
                logger.info('_direct_inference.')
                _direct_inference(bert_crf_tagger, tokenizer, infer_dataloader,
                                  index_to_label, vocab_dict, paras)

    logger.info(f'Train complete, Best F1-score: {best_eval["f1"]:.4f}.')



if __name__ == '__main__':

    args = args()

    set_seed(args.seed)

    if args.mode.lower() == 'train':
        main(args)
    elif args.mode.lower() == 'infer':
        inference(args)
    else:
        logger.warning(f'mode must be "train" or "infer", but get "{args.mode}"')
