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


# {'价格': 1272, '配置': 852, '操控': 1035, '舒适性': 930, '油耗': 1081, '动力': 2731, '内饰': 535, '安全性': 572, '空间': 441, '外观': 488}
SUBJECT_MASK = {'价格': 0, '配置': 1, '操控': 2, '舒适性': 3, '油耗': 4, '动力': 5, '内饰': 6, '安全性': 7, '空间': 8, '外观': 9}
class My_sentiment_dataset(data.Dataset):
    
    def __init__(self, max_len, cv=False, augment=False):
        train = pd.read_csv('data/train.csv')
        
        self.word = []
        self.characters = []
        self.topics = []
        self.label = train['sentiment_value']
        self.topics_indexs = [[] for _ in range(10)]    # 各主题样本的下标
        self.current_topic = 0  # 当前主题
        for i, content in enumerate(train['content']):
            characters_list = ''
            content = content.strip()
            content = re.sub(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', content)
            content = content.replace(' ', '')
            content = content.replace('，', ',')
            content = content.replace('？', '?')
            content = content.replace('！', '!')
            content = content.replace('（', '(')
            content = content.replace('）', ')')
            for c in content:
                characters_list += c
            content = jieba.lcut(content, cut_all=False)
            
            self.word.append(content)
            self.characters.append(characters_list)
            this_topic = SUBJECT_MASK[train['subject'][i]]
            self.topics_indexs[this_topic].append(i)
            self.topics.append(this_topic)

        self.max_len = max_len
        self.char_max_len = 150 # 119 %95
        self.augment = augment
        self.data_len = len(train)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=19950717)
        self.folds = []
        # 按主题均匀划分五折
        for train_index, test_index in skf.split(np.arange(self.data_len), np.array(self.topics)):
            self.folds.append((train_index, test_index))
        self.fold_index = 0
        self.current_index_set = self.folds[self.fold_index][0]
        if cv == False:
            self.current_index_set = range(self.data_len)
        self.trainning = True

        self.current_topic_index_set = My_sentiment_dataset.intersect(self.topics_indexs[self.current_topic], self.current_index_set)

        self.word2index = opt.word2index
        fff = open('char2index.json', 'r')
        self.char2index = json.load(fff)
        fff.close()

        random.seed(19950717)
    
    def get_len(self):
        return self.data_len

    def change_fold(self, fold_index):      # including change to training data
        self.fold_index = fold_index
        self.current_index_set = self.folds[self.fold_index][0]
        self.current_topic_index_set = My_sentiment_dataset.intersect(self.topics_indexs[self.current_topic], self.current_index_set)
        self.trainning = True
    
    def change2val(self):
        self.current_index_set = self.folds[self.fold_index][1]
        self.current_topic_index_set = self.current_index_set
        self.trainning = False
    
    def change2train(self):
        self.current_index_set = self.folds[self.fold_index][0]
        self.current_topic_index_set = My_sentiment_dataset.intersect(self.topics_indexs[self.current_topic], self.current_index_set)
        self.trainning = True
    
    def change_topic(self, topic):
        self.current_topic = topic
        self.current_topic_index_set = My_sentiment_dataset.intersect(self.topics_indexs[self.current_topic], self.current_index_set)
    
    @staticmethod
    def intersect(a, b):
        return list(set(a) & set(b))


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
        sentence = self.word[self.current_topic_index_set[index]]
        characters = self.characters[self.current_topic_index_set[index]]
        label = self.label[self.current_topic_index_set[index]]
        topic = self.topics[self.current_topic_index_set[index]]

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
        label = torch.from_numpy(np.array([label+1]))
        return sentence, characters, label, topic

    def __len__(self):
        return len(self.current_topic_index_set)

if __name__ == '__main__':
    #mmm = My_dataset(100)
    #mmm.multilabelfile('multilabel.csv')
    # out_file = open('test_processed.csv', 'w')
    # out_str = "%s,%s,%s\n" % ('id', 'article', 'word_seg')
    # fff = open('../word2index.json', 'r')
    # word2index = json.load(fff)
    # fff.close()
    # fff = open('../char2index.json', 'r')
    # char2index = json.load(fff)
    # fff.close()
    # test_file = pd.read_csv('test_public.csv')
    # for i in range(len(test_file['content'])):
    #     article = ''
    #     for j in test_file['content'][i]:
    #         article += str(char2index[j]) + ' '
    #     article = article[:len(article)-1]
    #     word_seg = ''
    #     for j in jieba.lcut(test_file['content'][i], cut_all=False):
    #         if word2index.get(j) is not None:
    #             word_seg += str(word2index[j]) + ' '
    #     word_seg = word_seg[:len(word_seg)-1]
    #     out_str += "%s,%s,%s\n" % (test_file['content_id'][i], article, word_seg)
    # out_file.write(out_str)
    # out_file.close()
    pass