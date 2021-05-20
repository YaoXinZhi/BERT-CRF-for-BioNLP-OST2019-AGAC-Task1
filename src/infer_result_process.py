# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 11/05/2021 9:05
@Author: XINZHI YAO
"""

# """
# Biomarkers	B-NegReg
# of	O
# Alzheimer	B-Disease
# 's	I-Disease
# disease	I-Disease
# among	O
# Mexican	O
# Americans	O
# .	O
#
# """
#
#
#
# data_list = ['Biomarkers', 'of', 'Alzheimer', "'s", 'disease',
#              'among', 'Mexican', 'Americans', '.']
#
# label_list = ['B-NegReg', 'O', 'B-Disease', 'I-Disease', 'I-Disease',
#               'O', 'O', 'O', 'O']

from itertools import permutations

def read_sentence_to_idx(sentence_idx_file: str):

    idx2sentence = {}
    with open(sentence_idx_file) as f:
        f.readline()
        for line in f:
            idx, pmid, sentence = line.strip().split('\t')
            idx2sentence[idx] = sentence
    return idx2sentence


def tag_process(data_list: list, label_list: list):
    """
    :param data_list: list of tokens.
    :param label_list: list of labels.
    :return: tag_set, sentence
    """
    sentence = ' '.join(data_list)

    start = 0
    tag_set = set()
    phrase_list = []
    for idx, token in enumerate(data_list):

        token_start = sentence.find(token, start)
        token_end = token_start + len(token)
        start = token_end
        # print(token, sentence[token_start: token_end])
        label = label_list[idx]
        if label == 'O' or label.startswith('I-'):
            continue
        if label.startswith('B-'):

            phrase_list = []
            offset_list = []
            tag = label.split('-')[1]
            phrase_list.append(token)
            offset_list.append((token_start, token_end))
            _start = 0
            for _idx, _token in enumerate(data_list[idx+1:]):

                _idx = _idx + idx + 1
                _label = label_list[_idx]

                if len(_label.split('-')) > 1:
                    _tag = _label.split('-')[1]
                else:
                    _tag = ''

                if _label == 'O' or _tag != tag:
                    #todo add
                    phrase = ' '.join(phrase_list)
                    offset = (offset_list[0][0], offset_list[-1][1])
                    tag_set.add((phrase, tag, offset))
                    break

                _token_start = sentence.find(_token, _start)
                _token_end = _token_start + len(_token)
                phrase_list.append(_token)
                offset_list.append((_token_start, _token_end))
    return tag_set, sentence


def tagging_to_re_input_wordpiece(data_list: list, label_list: list):
    data_list = ['Evidence', ]




def tagging_to_re_input(infer_result: str, save_file: str):

    wf = open(save_file, 'w')
    wf.write(f'Token1\tLabel1\tOffset1\t'
             f'Token2\tLabel2\tOffset2\t'
             f'Relation\tSentence\n')
    data_list = []
    label_list = []

    save_count = 0
    with open(infer_result) as f:
        for line in f:
            l = line.strip().split('\t')
            if l == ['']:
                # todo: I- Kai tou de zhu shi
                has_label = False
                for label in label_list:
                    if label_list != 'O':
                        has_label = True

                if not has_label:
                    continue

                # fixme: add pubtator tagging
                tag_set, sentence = tag_process(data_list, label_list)

                data_list = []
                label_list = []

                # save RelationExtract input format
                if len(tag_set) < 2:
                    continue
                for (tag1, tag2) in permutations(tag_set, 2):

                    token1, label1, offset1 = tag1
                    token2, label2, offset2 = tag2
                    save_count += 1
                    wf.write(f'{token1}\t{label1}\t{offset1}\t'
                             f'{token2}\t{label2}\t{offset2}\t'
                             f'None\t{sentence}\n')
            else:
                if len(l) < 2:
                    continue
                data_list.append(l[0])
                label_list.append(l[1])

    wf.close()
    print(f'{save_count:,} evidence saved.')
    print(f'{save_file} save done.')

if __name__ == '__main__':

    # infer_result_file = '../infer_result/infer_output.0511.max128.txt'
    # re_input_file = '../infer_result/ad.RE.txt'

    word_piece = True

    infer_result_file = '../infer_result/ad.word-piece.infer_output.cp.txt'
    sentence_mapping_file = '../data/pubtator_file/ad.sent.txt'

    re_input_file = '../infer_result/RE_input/ad.word-piece.RE.txt'

    print('Running.')
    print('reading sentence_idx_file.')
    idx_to_sentence = read_sentence_to_idx(sentence_mapping_file)
    print('tagging to RE input.')
    if not word_piece:
        tagging_to_re_input(infer_result_file, re_input_file)
    else:
        pass

