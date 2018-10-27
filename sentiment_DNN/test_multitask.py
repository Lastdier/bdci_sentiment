from torch.utils import data
import pandas as pd
import jieba
import random
import numpy as np
import json
import torch
import sys
import models
import tqdm
from config import opt
import re
from data.multitask_dataset import My_sentiment_dataset
from sklearn.externals import joblib
import fire
from sklearn.metrics import f1_score
sys.path.append("..")


weight_file = open('weights.csv', 'w')
SUBJECT_LIST = ['价格', '配置', '操控', '舒适性', '油耗', '动力', '内饰', '安全性', '空间', '外观']
SUBJECT_MASK = {'价格': 0, '配置': 1, '操控': 2, '舒适性': 3, '油耗': 4, '动力': 5, '内饰': 6, '安全性': 7, '空间': 8, '外观': 9}
class My_test_dataset(data.Dataset):
    
    def __init__(self, max_len):
        test = pd.read_csv('data/test_public.csv')
        content_dict = {}
        topics = pd.read_csv('rcnn.csv')
        self.content_id = topics['content_id']
        self.subject = topics['subject']
        for i, con_id in enumerate(test['content_id']):
            content_dict[con_id] = test['content'][i]
        self.content = []
        for con_id in self.content_id:
            self.content.append(content_dict[con_id])


        self.max_len = max_len
        self.data_len = len(self.content_id)

        fff = open('word2index.json', 'r')
        self.word2index = json.load(fff)
        fff.close()
        fff = open('char2index.json', 'r')
        self.char2index = json.load(fff)
        fff.close()
        self.char_max_len = 150

    def __getitem__(self, index):
        sen_id = self.content_id[index]
        sentence = self.content[index]
        topic = self.subject[index]
        sentence = sentence.strip()
        sentence = sentence.replace(' ', '')
        sentence = sentence.replace('，', ',')
        sentence = sentence.replace('？', '?')
        sentence = sentence.replace('！', '!')
        sentence = sentence.replace('（', '(')
        sentence = sentence.replace('）', ')')
        sentence = re.sub(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', sentence)
        sentence = jieba.lcut(sentence, cut_all=False)

        # sentence = [self.word2index[word] for word in sentence if self.word2index.get(word) is not None else self.word2index['unknown']]
        sen_inds = []
        characters = ''
        for word in sentence:
            if word == ' ':
                continue
            characters += word
            if self.word2index.get(word) is not None:
                sen_inds.append(self.word2index[word])
            else:
                sen_inds.append(self.word2index['<oov>'])
        
        char_inds = []
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

        characters = torch.from_numpy(np.array(char_inds)).long()
        sentence = torch.from_numpy(np.array(sen_inds)).long()
        return sentence, characters, sen_id, topic

    def __len__(self):
        return self.data_len


def testing(name=None, model=None, get_prob=False):
    lll = 100
    if opt.type_ == "char":
        lll = 150
    dataset = My_test_dataset(lll)
    dataloader = data.DataLoader(dataset,
                batch_size=128,
                shuffle=False,
                num_workers=4,
                pin_memory=True
    )

    if model is None:
        model = getattr(models,opt.model)(opt).cuda()
        state = torch.load(name+'.pt')
        model.load_state_dict(state)

    preds = []
    pred_probs = []
    for i, (content, characters, sen_id, topic) in tqdm.tqdm(enumerate(dataloader)):
        # 训练 更新参数
        if opt.type_ == 'word':
            content = content.cuda()
        elif opt.type_ == 'char':
            content = characters.cuda()

        score = model(content, topic)

        pred_prob = torch.nn.functional.softmax(score).detach().cpu().tolist()
        # pred_prob = score.detach().cpu().tolist()
        
        for ppp in pred_prob:
            pred_probs.append(ppp)

        predict = score.data.topk(1,dim=1)[1].cpu().tolist()
        for ppp in predict:
            preds.append(SUBJECT_LIST[ppp[0]])

    if get_prob:
        return np.array(pred_probs)

    test_id = pd.read_csv('data/test_public.csv')[["content_id"]].copy()

    #保存概率文件
    test_prob=pd.DataFrame(pred_probs)
    test_prob.columns=["class_prob_%s"%i for i in range(10)]
    test_prob["content_id"]=list(test_id["content_id"])
    test_prob.to_csv('result/prob_'+name+'.csv',index=None)

    
    test_pred=pd.DataFrame(preds)
    test_pred.columns=["subject"]
    test_pred["subject"]=(test_pred["subject"])
    print(test_pred.shape)
    print(test_id.shape)
    test_pred["content_id"]=list(test_id["content_id"])
    test_pred["sentiment_value"]=np.zeros((len(test_id["content_id"])), dtype=np.int32)
    test_pred["sentiment_word"]=np.nan
    test_pred[["content_id","subject","sentiment_values","sentiment_word"]].to_csv('result/sub_'+name+'.csv',index=None)


def val_fold(name, dataset, pred_probs, train_predict, true_target):
    '''
    计算模型在验证集上的分数
    '''
    model = getattr(models,opt.model)(opt).cuda()
    print(model)
    state = torch.load(name+'.pt')
    model.load_state_dict(state)

    dataset.change2val()
    dataloader = data.DataLoader(dataset,
                    batch_size = 1,
                    shuffle = False,
                    num_workers = opt.num_workers,
                    pin_memory = True
                    )
    
    with torch.no_grad():
        for content,characters,label,topic, sen_id in dataloader:
            sen_id = sen_id.tolist()
            if opt.type_ == 'word':
                content= content.cuda()
            elif opt.type_ == 'char':
                content = characters.cuda()
            score = model(content, topic)
            # predict = score.detach().cpu().numpy()
            # predict_ind = np.zeros((predict.shape[0], 10), dtype=np.int32)
            # for i in range(predict.shape[0]):
            #     ttt = predict[i]
            #     tttt = [ttt>0.]
            #     predict_ind[i][tttt] = 1
            pred_prob = torch.nn.functional.softmax(score).detach().cpu().numpy()
            predict = score.data.topk(1,dim=1)[1].cpu().tolist()
            target = label.tolist()
            train_predict += predict
            true_target += target
            
            # for i, ppp in enumerate(pred_prob):
            #     pred_probs[sen_id[i]] = ppp


    del score

    # model.train()   #???
    return pred_probs, np.zeros((1,1)), train_predict, true_target


def stacking_train_set(**kwargs):
    opt.parse(kwargs, print_=True)
    lll = 100
    if opt.type_ == "char":
        lll = 150
    dataset = My_sentiment_dataset(lll, cv=True)
    cv0 = 'Multitask_0_score0.663224781573'
    cv1 = 'Multitask_1_score0.691971383148'
    cv2 = 'Multitask_2_score0.70445505171'
    cv3 = 'Multitask_3_score0.681782020684'
    cv4 = 'Multitask_4_score0.686055776892'

    pred_probs = {}
    train_predict = []
    true_target = []
    pred_probs, test_prob, train_predict, true_target = val_fold(cv0, dataset, pred_probs, train_predict, true_target)
    x_test = np.zeros(test_prob.shape)
    x_test += test_prob
    dataset.change_fold(1)

    pred_probs, test_prob, train_predict, true_target = val_fold(cv1, dataset, pred_probs, train_predict, true_target)
    x_test += test_prob
    dataset.change_fold(2)

    pred_probs, test_prob, train_predict, true_target = val_fold(cv2, dataset, pred_probs, train_predict, true_target)
    x_test += test_prob
    dataset.change_fold(3)

    pred_probs, test_prob, train_predict, true_target = val_fold(cv3, dataset, pred_probs, train_predict, true_target)
    x_test += test_prob
    dataset.change_fold(4)

    pred_probs, test_prob, train_predict, true_target = val_fold(cv4, dataset, pred_probs, train_predict, true_target)
    x_test += test_prob

    f1 = f1_score(true_target, train_predict, average='micro')
    print('train micro f1:' + str(f1))
    # train_id = pd.read_csv('data/multilabel.csv')['id']
    # xxx = []
    # for i in train_id:
    #     xxx.append(pred_probs[i])
    # xxx = np.array(xxx)
    # print(xxx.shape)

    #x_test /= 5
    #joblib.dump((xxx, x_test), 'rcnn_char.pk')


if __name__ == "__main__":
    fire.Fire()