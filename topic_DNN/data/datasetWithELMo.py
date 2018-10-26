from torch.utils import data
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import random
import numpy as np
import json
import torch
import sys
sys.path.append("..")
from config import opt
from sklearn.externals import joblib


# {'价格': 1272, '配置': 852, '操控': 1035, '舒适性': 930, '油耗': 1081, '动力': 2731, '内饰': 535, '安全性': 572, '空间': 441, '外观': 488}
SUBJECT_MASK = {'价格': 0, '配置': 1, '操控': 2, '舒适性': 3, '油耗': 4, '动力': 5, '内饰': 6, '安全性': 7, '空间': 8, '外观': 9}
class My_dataset(data.Dataset):
    
    def __init__(self, cv=False):    
        self.elmo = joblib.load('data/sample_vector.pk')
        # (8290, 100, 1024)
        self.index2id = []
        self.train_no_dup = {}
        train = pd.read_csv('data/train.csv')
        for i, content_id in enumerate(train['content_id']):
            # create label
            if self.train_no_dup.get(content_id) is None:
                self.index2id.append(content_id)
                label = [0] * 10
                label[SUBJECT_MASK[train['subject'][i]]] = 1   

                self.train_no_dup[content_id] = label
            # add new label
            else:
                self.train_no_dup[content_id][SUBJECT_MASK[train['subject'][i]]] = 1

        del train

        self.seq_len = 100
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

        random.seed(19950717)

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

    def __getitem__(self, index):
        sen_id = self.index2id[self.current_index_set[index]]
        label = self.train_no_dup[sen_id]

        sentence = torch.from_numpy(self.elmo[self.current_index_set[index]])
        label = torch.from_numpy(np.array(label)).float()
        return sentence, label, sen_id

    def __len__(self):
        return len(self.current_index_set)

if __name__ == '__main__':
    mmm = My_dataset(100)
    mmm.multilabelfile('multilabel_n.csv')
    pass