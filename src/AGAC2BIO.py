# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 09:35:15 2019

@author: wyx
"""

import json
from glob import glob
import spacy
import en_core_web_sm
from tqdm import tqdm

#nlp = spacy.load('en')
nlp = en_core_web_sm.load()

def json2bio(fpath,output,splitby = 's'):
    '''
    输入json文件，返回bio(token pmid label)
    splitby = 's' ----以句子空格
    splitby = 'a' ----以摘要空格
    '''
    with open(fpath) as f:
        pmid = fpath[-13:-5]
        annotations = json.load(f)
        text = annotations['text'].replace('\n',' ')
        all_words = text.split(' ')
        all_words2 = [token for token in nlp(text)]
        all_label = ['O']*len(all_words)
        for i in annotations['denotations']:
            b_location = i['span']['begin']
            e_location = i['span']['end']
            label = i['obj']
            B_wordloc = text.count(' ',0,b_location)
            I_wordloc = text.count(' ',0,e_location)
            all_label[B_wordloc] = 'B-'+label
            if B_wordloc != I_wordloc:
                for word in range(B_wordloc+1,I_wordloc+1):
                    all_label[word] = 'I-'+label

        #得到以空格分词的词列表和对应标签列表
        for w,_ in enumerate(all_words):
            all_words[w] = nlp(all_words[w])
        #len(all_words)=271
        #对单个元素分词 
        labelchange = []
        for i,_ in enumerate(all_words):
            token = [token for token in all_words[i]]
            if len(token)==1:
                labelchange.append(all_label[i])
            else:
                if all_label[i] == 'O':
                    labelchange.extend(['O']*len(token))
                if all_label[i] != 'O':
                    labelchange.append(all_label[i])
                    if str(token[-1]) == '.' or str(token[-1]) == ',':
                        labelchange.extend(['I-'+all_label[i][2:]]*(len(token)-2))
                        labelchange.append('O')
                    else:
                        labelchange.extend(['I-'+all_label[i][2:]]*(len(token)-1))
        #写入文件
        with open(output,'a',encoding='utf-8') as f:
            #以句子空行
            if splitby == 's':
                for j,_ in enumerate(all_words2):
                    if str(all_words2[j]) == '.' and str(all_words2[j-1]) != 'p':
                        #line =str(all_words2[j])+'\t'+pmid+'\t'+labelchange[j]+'\n'
                        try:
                            line = str(all_words2[j]) + '\t' + '\t' + labelchange[j] + '\n'
                        except:
                            print(line)
                            continue

                        f.write(line+'!!!\tO\n')
                    else:
                        #line =str(all_words2[j])+'\t'+pmid+'\t'+labelchange[j]+'\n'
                        line = str(all_words2[j]) + '\t' + '\t' + labelchange[j] + '\n'
                        f.write(line)
            #以摘要空行
            if splitby == 'a':
                for j,_ in enumerate(all_words2):
                    #line =str(all_words2[j])+'\t'+pmid+'\t'+labelchange[j]+'\n'
                    line = str(all_words2[j]) + '\t' + '\t' + labelchange[j] + '\n'
                    f.write(line)
                f.write('\n')
            #f.write('c617666\n')


if __name__ == "__main__":
    AGAC_json_path = '../data/AGAC_data/AGAC_v3/annotations/json/*.json'
    save_file = '../data/train.agac.v3.txt'

    fpathlist = glob(AGAC_json_path)
    # fpathlist = glob('AGAC_training/json/*.json')
    #fpathlist = glob('AGAC_answer/json/*.json')
    # output = "train_BIO.txt"
    #output = "test_BIO.txt"
    print('Running')
    for i in tqdm(fpathlist):
        json2bio(i,save_file,'s')


