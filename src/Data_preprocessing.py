# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 18/05/2021 10:31
@Author: XINZHI YAO
"""

import re
import os

from nltk import word_tokenize, sent_tokenize

from collections import defaultdict

from nltk import tokenize

# from utils import *

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

def denotation_sent_map(denotation_set: set, sent_to_offset: dict):

    sent_to_denotation = defaultdict(set)

    for sentence, (s_start, s_end) in sent_to_offset.items():
        for (mention, _type, _id, (t_start, t_end)) in denotation_set:

            if t_start >= s_start and t_end <= s_end:
                # find token offset in sentence
                start = 0
                while 1:
                    new_t_start = sentence.find(mention, start)
                    if new_t_start == -1:
                        break
                    new_t_end = new_t_start + len(mention)

                    if sentence[new_t_start: new_t_end] != mention:
                        print('Warning: Position calculation error.')
                    sent_to_denotation[sentence].add((mention, _type, _id, (new_t_start, new_t_end)))
                    start = new_t_end
    return sent_to_denotation

def get_token_offset(sentence: str):
    token_to_offset = {}

    start = 0
    for token in word_tokenize(sentence):

        token_start = sentence.find(token, start)
        token_end = token_start + len(token)
        token_to_offset[token] = (token_start, token_end)
    return token_to_offset


def pubtator_to_bio(pubtator_file: str, save_path: str, prefix: str='pubtator'):

    bio_save_file = os.path.join(save_path, f'{prefix}.bio.txt')
    sent_save_file = os.path.join(save_path, f'{prefix}.sent.txt')

    wf = open(bio_save_file , 'w', encoding='utf-8')
    wf_sent = open(sent_save_file, 'w', encoding='utf-8')
    wf_sent.write('SentenceIndex\tPMID\tSentence\n')


    doc = ''
    sent_to_idx = {}
    processed_count = 0
    denotation_set = set()
    with open(pubtator_file) as f:
        for line in f:
            l = line.strip()
            if '|t|' in l or '|a|' in l:
                pmid, text_type, text = l.split('|')
                if text_type == 't':
                    processed_count += 1
                    doc = ''
                    doc += text

                    if processed_count % 500 == 0:
                        print(f'{processed_count} pubtator processed.')

                elif text_type == 'a':
                    doc += f' {text}'
                continue
            l_split = l.split('\t')
            if l_split == ['']:

                sent_to_offset, sent_to_id = get_sent_offset(doc)
                sent_to_denotation = denotation_sent_map(denotation_set, sent_to_offset)

                for sentence, denotation_set in sent_to_denotation.items():

                    if sent_to_idx.get(sentence):
                        sentence_idx = sent_to_idx[sentence]
                    else:
                        sentence_idx = len(sent_to_idx)
                        sent_to_idx[sentence] = sentence_idx


                    wf_sent.write(f'{sentence_idx}\t{pmid}\t{sentence}\n')
                    token_offset = get_token_offset(sentence)
                    for token, (t_start, t_end) in token_offset.items():
                        save_flag = False
                        for mention, _type, _id, (d_start, d_end) in denotation_set:
                            if t_start >= d_start and t_end <= d_end:
                                save_flag = True
                                wf.write(f'{sentence_idx}\t{pmid}\t{token}\t{_type}\t{_id}\n')
                        if not save_flag:
                            wf.write(f'{sentence_idx}\t{pmid}\t{token}\tO\tNone\n')
                    wf.write('\n')

                denotation_set = set()
            else:
                try:
                    _, start, end, mention, _type, _id = l_split
                except:
                    _, start, end, mention, _type = l_split
                    _id = 'None'
                start, end = int(start), int(end)
                denotation_set.add((mention, _type, _id, (start, end)))

                # check the denotation
                if doc[int(start): int(end)] != mention:
                    print('Warning: Token position calculation error.')
                    print(pmid)
                    print(doc)
                    print(f'{start}, {end}')
                    print(f'{doc.find(mention)}, {doc.find(mention)+len(mention)}')
                    print(f'{doc[int(start): int(end)]}, mention: {mention}')
    wf.close()
    wf_sent.close()
    print(f'{bio_save_file} save done.')
    print(f'{sent_save_file} save done.')

if __name__ == '__main__':

    ad_pubtator_file = '../data/pubtator_file/ad.pubtator_central.txt'

    save_path = '../data/pubtator_file'
    save_prefix = 'ad'

    ad_bio_file = '../data/pubtator_file/ad.pubtator.infer.txt'

    pubtator_to_bio(ad_pubtator_file, save_path, save_prefix)

