# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 31/03/2021 9:39
@Author: XINZHI YAO
"""

import os
import logging


import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import AdamW, WarmUp
from transformers import BertTokenizerFast, BertModel, BertTokenizer

# todo: tensorboardX
from tensorboardX import SummaryWriter

from src.model import BERT_CRF
from src.utils import *
from src.dataloader import SeqLabeling_Dataset
from src.config import args


# todo
def evaluation(paras, model, tokenizer, idx_to_label: dict, ):

    model.eval()

    test_dataset = SeqLabeling_Dataset(paras.test_data, paras.label_file)
    label_to_index = test_dataset.label_to_index

    test_dataloader = DataLoader(test_dataset, batch_size=paras.batch_size,
                            shuffle=False, drop_last=paras.drop_last)

    ori_label_list = []
    pre_label_list = []
    for step, batch in enumerate(test_dataloader):
        batch_data, batch_label = batch

        batch_data_list = [data.split('&&&') for data in batch_data]
        batch_label_list = [label.split('&&&') for label in batch_label]
        encoded_input = tokenizer(batch_data_list,
                                  return_offsets_mapping=True,
                                  max_length=paras.max_length,
                                  truncation=True,
                                  is_split_into_words=True,
                                  padding=True,
                                  return_tensors='pt').to(device)
        batch_offset_mapping = encoded_input['offset_mapping']

        batch_attention_mask = encoded_input['attention_mask']

        batch_adjust_label_list = batch_adjust_label(batch_offset_mapping,
                                                     batch_label_list)
        batch_ori_label_list = []
        for idx, mask in enumerate(batch_attention_mask):
            print(idx, len(tensor_to_list(mask)), mask.sum())
            batch_ori_label_list.append(batch_adjust_label_list[idx][:mask.sum()])

        predict_result = model.predict(encoded_input)
        predict_label_list = []
        for pred_label in predict_result:
            predict_label_list.append([idx_to_label[label] for label in pred_label])

        ori_label_list.extend(batch_label_idx)
        pre_label_list.extend(predict_label_list)

    f1 = f1_score(ori_label_list, pre_label_list)

    acc = accuracy_score(ori_label_list, predict_label_list)

    precision = precision_score(ori_label_list, predict_label_list)

    recall = recall_score(ori_label_list, predict_label_list)



    model.train()
    return f1, acc, precision, recall

def main(paras):


    logger = logging.getLogger(__name__)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.DEBUG)

    # writer = SummaryWriter(logdir=os.path.join(args.output_dir, "eval"), comment="Linear")


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = SeqLabeling_Dataset(paras.train_data, paras.label_file)
    label_to_index = train_dataset.label_to_index

    train_dataloader = DataLoader(train_dataset, batch_size=paras.batch_size,
                                  shuffle=paras.shuffle, drop_last=paras.drop_last)

    # load model
    print(f'Loading model: {paras.model_name}.')
    tokenizer = BertTokenizerFast.from_pretrained(paras.model_name)
    model = BERT_CRF(paras)

    # if paras.optimizer == 'sgd':
    #     optimizer = optim.SGD(model.parameters(), lr=paras.learning_rate, momentum=paras.momentum)
    # elif paras.optimizer == 'adam':
    #     optimizer = optim.Adam(model.parameters(), lr=paras.learning_rate, weight_decay=paras.weight_decay)
    # else:
    #     raise ValueError('optimizer not supported.')

    if paras.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = paras.max_steps // (len(train_dataloader) // paras.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // paras.gradient_accumulation_steps * paras.num_train_epochs


    no_decay = ['bias', 'layerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [ p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) ],
         'weight_decay': 0.01},
        {'params': [ p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) ],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=paras.learning_rate,
                      eps=paras.adam_epsilon)

    # todo: transformers WarmupLinearSchedule

    logger.info('****** Running training ******')
    logger.info(f' Num Epochs: {paras.num_train_epochs}')
    logger.info(f' Total optimization steps: {t_total}')

    model.train()
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_f1 = 0.0
    # start to train.

    for step, batch in enumerate(train_dataloader):

        batch_data, batch_label = batch

        batch_data_list = [data.split('&&&') for data in batch_data]
        batch_label_list = [label.split('&&&') for label in batch_label]

        encoded_input = tokenizer(batch_data_list,
                                  return_offsets_mapping=True,
                                  max_length=paras.max_length,
                                  truncation=True,
                                  is_split_into_words=True,
                                  padding=True,
                                  return_tensors='pt').to(device)

        # test
        # input_ids = encoded_input['input_ids']
        # token_type_ids = encoded_input['token_type_ids']
        # attention_mask = encoded_input['attention_mask']
        # offset_mapping = encoded_input['offset_mapping']

        batch_offset_mapping = encoded_input['offset_mapping']

        batch_adjust_label_list = batch_adjust_label(batch_offset_mapping,
                                                     batch_label_list)

        # for i in range(len(batch_offset_mapping)):
        #     # print(len(batch_offset_mapping[i]), len(batch_adjust_label_list[i]), len(input_ids[i]))
        #     for j in range(len(batch_offset_mapping[ i ])):
        #         print(f'{tokenizer.decode(input_ids[ i ][ j ])}, {batch_adjust_label_list[ i ][ j ]}')

        batch_label_idx = []
        for label_list in batch_adjust_label_list:
            batch_label_idx.append([label_to_index[label] for label in label_list])

        # batch_input_ids = torch.LongTensor(encoded_input['input_ids']).to(device)
        # batch_token_type_ids = torch.LongTensor(encoded_input['token_type_ids'])
        batch_label_idx = torch.LongTensor(batch_label_idx).to(device)
        # batch_attention_mask = torch.LongTensor(encoded_input['attention_mask']).to(device)

        # todo: acc, example
        loss = model(encoded_input, batch_label_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.info(f'Step:{step}, loss:{loss.item()}')


        # todo: n_gpu

        # check
        # if args.gradient_accumulation_steps > 1:
        #     loss = loss / args.gradient_accumulation_steps
        #
        # # todo: gradient accumulation
        # loss.backward()
        # tr_loss += loss.item()
        # if (step + 1) % paras.gradient_accumulation_steps == 0:
        #     optimizer.step()
        #     # scheduler.step()
        #     model.zero_gred()
        #     global_step += 1
        #
        #     if args.logging_steps > 0 and global_step % paras.logging_steps == 0:
        #         tr_loss_avg = (tr_loss- logging_loss)/paras.logging_steps
        #         writer.add_scalar("Train/loss", tr_loss_avg, global_step)
        #         logging_loss = tr_loss_avg


    wf.close()



        # evaluation
        # predict_result = model.predict(encoded_input)


        # todo: loss log.
        # todo: evaluation


if __name__ == '__main__':


    paras = args()


    main(paras)




