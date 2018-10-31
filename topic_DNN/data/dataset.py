from torch.utils import data
import pandas as pd
import jieba
from sklearn.model_selection import StratifiedKFold
import random
import numpy as np
import json
import torch
import sys
sys.path.append("..")
from config import opt
import re
from pyhanlp import *


jieba.load_userdict('userdict.txt')
# {'价格': 1272, '配置': 852, '操控': 1035, '舒适性': 930, '油耗': 1081, '动力': 2731, '内饰': 535, '安全性': 572, '空间': 441, '外观': 488}
SUBJECT_MASK = {'价格': 0, '配置': 1, '操控': 2, '舒适性': 3, '油耗': 4, '动力': 5, '内饰': 6, '安全性': 7, '空间': 8, '外观': 9}
class My_dataset(data.Dataset):
    
    def __init__(self, max_len, cv=False, augment=False):
        train = pd.read_csv('data/train.csv')
        self.train_no_dup = {}  # [content_id][0] content  [content_id][1] multi-label
        self.index2id = []
        for i, content_id in enumerate(train['content_id']):
            # create label
            if self.train_no_dup.get(content_id) is None:
                self.index2id.append(content_id)
                label = [0] * 10
                label[SUBJECT_MASK[train['subject'][i]]] = 1
                sentiment = [0] * 3
                sentiment[train['sentiment_value'][i] + 1] = 1    # 情感值+1
                # 使用jieba分词
                content = train['content'][i].strip()
                content = re.sub(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', content)
                content = content.replace(' ', '')
                content = content.replace('，', ',')
                content = content.replace('？', '?')
                content = content.replace('！', '!')
                content = content.replace('（', '(')
                content = content.replace('）', ')')
                # content = jieba.lcut(content, cut_all=False)
                character_list = ''
                for w in HanLP.segment(content):
                    character_list += w.word   

                self.train_no_dup[content_id] = [content, label, character_list, sentiment]
            # add new label
            else:
                self.train_no_dup[content_id][1][SUBJECT_MASK[train['subject'][i]]] = 1
                self.train_no_dup[content_id][3][train['sentiment_value'][i] + 1] += 1

        self.max_len = max_len
        self.char_max_len = 150 # 119 %95
        self.augment = augment
        self.data_len = len(self.train_no_dup)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=19950717)
        self.folds = []
        for train_index, test_index in skf.split(np.arange(self.data_len), np.zeros((self.data_len,))):
            self.folds.append((train_index, test_index))
        self.fold_index = 0
        self.current_index_set = self.folds[self.fold_index][0]
        if cv == False:
            self.current_index_set = range(self.data_len)
        self.trainning = True

        self.word2index = opt.word2index
        fff = open('char2index.json', 'r')
        self.char2index = json.load(fff)
        fff.close()

        random.seed(19950717)
    
    def multilabelfile(self, path, sentiment=False):
        out_file = open(path, 'w', encoding='utf-8')
        out_str = "%s,%s,%s,%s\n" % ('id', 'article', 'word_seg','class')
        
        for i in self.train_no_dup:
            temp = ''
            if sentiment:   
                # 判断有无情感
                # if self.train_no_dup[i][3][0] == 0 and self.train_no_dup[i][3][2] == 0:
                #     temp += str(0)
                # else:
                #     temp += str(1)

                # 判断情感三分类
                # if self.train_no_dup[i][3][1] > 0:
                #     temp += str(1)
                # elif self.train_no_dup[i][3][0] > self.train_no_dup[i][3][2]:
                #     temp += str(0)
                # elif self.train_no_dup[i][3][2] > self.train_no_dup[i][3][0]:
                #     temp += str(2)
                # else:
                #     temp += str(0)

                # 判断正负情感
                if self.train_no_dup[i][3][0] > self.train_no_dup[i][3][2]:
                    temp += str(0)  # 负情感为0
                elif self.train_no_dup[i][3][2] > self.train_no_dup[i][3][0]:
                    temp += str(1)  # 正情感为1
                else:
                    continue
            else:
                for j in range(10):
                    if self.train_no_dup[i][1][j] != 0:
                        temp += str(j) + ' '
                temp = temp[:len(temp)-1]
            article = ''
            for j in self.train_no_dup[i][2]:
                article += str(self.char2index[j]) + ' '
            article = article[:len(article)-1]
            word_seg = ''
            for j in self.train_no_dup[i][0]:
                if self.word2index.get(j) is not None:
                    word_seg += str(self.word2index[j]) + ' '
            word_seg = word_seg[:len(word_seg)-1]
            out_str += "%s,%s,%s,%s\n" % (i, article, word_seg, temp)
        out_file.write(out_str)
        out_file.close()

    def change_fold(self, fold_index):      # including change to training data
        self.fold_index = fold_index
        self.current_index_set = self.folds[self.fold_index][0]
        self.trainning = True
    
    def change2val(self):
        self.current_index_set = self.folds[self.fold_index][1]
        self.trainning = False
    
    def change2train(self):
        self.current_index_set = self.folds[self.fold_index][0]
        self.trainning = True

    # def to_index(self, word):
    #     if self.word2index.get(word) is None:
    #         return 498681
    #     else:
    #         return self.word2index[word]

    def dropout(self,d,p=0.2):
        nnn = []
        for i in d:
            if random.random() > p:
                nnn.append(i)
        return nnn

    def shuffle(self,d):
        return np.random.permutation(d)

    def __getitem__(self, index):
        
        sen_id = self.index2id[self.current_index_set[index]]
        sentence = self.train_no_dup[sen_id][0]
        characters = self.train_no_dup[sen_id][2]
        label = self.train_no_dup[sen_id][1]
        
        # sentence = jieba.lcut(sentence, cut_all=False) # 分词

        if self.augment and self.trainning:
            temp = random.random()

            if temp < 0.5:
                if opt.type_ == 'word':
                    sentence = self.dropout(sentence)
                elif opt.type_ == 'char':
                    characters = self.dropout(characters)
            # elif temp < 0.55:
            #     sentence = self.shuffle(sentence)

        # sentence = [self.word2index[word] for word in sentence if self.word2index.get(word) is not None else self.word2index['unknown']]
        sen_inds = []
        char_inds = []
        for word in sentence:
            if self.word2index.get(word) is not None:
                sen_inds.append(self.word2index[word])
            else:
                sen_inds.append(self.word2index['<oov>'])
        for char in characters:
            if self.char2index.get(char) is not None:
                char_inds.append(self.char2index[char])
            else:
                char_inds.append(self.char2index['<oov>'])

        if len(sen_inds) > self.max_len: 
            sen_inds = sen_inds[:self.max_len]
        else:
            pad = [0] * (self.max_len - len(sen_inds))
            sen_inds += pad
        
        if len(char_inds) > self.char_max_len: 
            char_inds = char_inds[:self.char_max_len]
        else:
            pad = [0] * (self.char_max_len - len(char_inds))
            char_inds += pad

        sentence = torch.from_numpy(np.array(sen_inds)).long()
        characters = torch.from_numpy(np.array(char_inds)).long()
        label = torch.from_numpy(np.array(label)).float()
        return sentence, characters, label, sen_id

    def __len__(self):
        return len(self.current_index_set)

if __name__ == '__main__':
    mmm = My_dataset(100)
    mmm.multilabelfile('multilabel_polarity.csv', sentiment=True)
    pass