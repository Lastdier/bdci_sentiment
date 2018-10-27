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
from data.datasetWithELMo import My_dataset
from sklearn.externals import joblib
import fire
sys.path.append("..")
from sklearn.externals import joblib


weight_file = open('weights.csv', 'w')
SUBJECT_LIST = ['价格', '配置', '操控', '舒适性', '油耗', '动力', '内饰', '安全性', '空间', '外观']
SUBJECT_MASK = {'价格': 0, '配置': 1, '操控': 2, '舒适性': 3, '油耗': 4, '动力': 5, '内饰': 6, '安全性': 7, '空间': 8, '外观': 9}
class My_test_dataset(data.Dataset):
    
    def __init__(self):
        self.elmo = joblib.load('data/sample_test_vector.pk')
        self.test = pd.read_csv('data/test_public.csv')

        self.data_len = self.elmo.shape(0)

    def __getitem__(self, index):
        sen_id = self.test['content_id'][index]
        return self.elmo[index], sen_id

    def __len__(self):
        return self.data_len


def testing(name=None, model=None, get_prob=False):
    dataset = My_test_dataset()
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
    for i, (content, sen_id) in tqdm.tqdm(enumerate(dataloader)):
        content = content.cuda()
        score = model(content)
        if opt.model == 'LSTMwithAtten':
            score = score[0]

        pred_prob = torch.nn.functional.sigmoid(score).detach().cpu().tolist()
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


def val_fold(name, dataset, pred_probs):
    '''
    计算模型在验证集上的分数
    '''
    model = getattr(models,opt.model)(opt).cuda()
    print(model)
    state = torch.load(name+'.pt')
    model.load_state_dict(state)

    dataset.change2val()
    dataloader = data.DataLoader(dataset,
                    batch_size = opt.batch_size,
                    shuffle = False,
                    num_workers = opt.num_workers,
                    pin_memory = True
                    )
    
    with torch.no_grad():
        for content,label,sen_id in dataloader:
            sen_id = sen_id.tolist()
            if opt.type_ == 'word':
                content,label = content.cuda(),label.cuda()
            score = model(content)
            if opt.model == 'LSTMwithAtten':
                weights = score[1].detach().cpu().numpy()
                score = score[0]
                for l, iid in enumerate(sen_id):
                    temp = iid + ' '
                    for wei in weights[l]:
                        temp += str(wei) + ' '
                    print(temp, file=weight_file)
            # predict = score.detach().cpu().numpy()
            # predict_ind = np.zeros((predict.shape[0], 10), dtype=np.int32)
            # for i in range(predict.shape[0]):
            #     ttt = predict[i]
            #     tttt = [ttt>0.]
            #     predict_ind[i][tttt] = 1
            pred_prob = torch.nn.functional.sigmoid(score).detach().cpu().numpy()
            # pred_prob = score.detach().cpu().numpy()
            # for i, ppp in enumerate(predict_ind):
            #     predicts.append((sen_id[i], ppp))
            
            for i, ppp in enumerate(pred_prob):
                pred_probs[sen_id[i]] = ppp

    del score

    # model.train()   #???
    return pred_probs, testing(model=model, get_prob=True)


def stacking_train_set(**kwargs):
    opt.parse(kwargs, print_=True)
    lll = 100
    if opt.type_ == "char":
        lll = 150
    dataset = My_dataset(cv=True)
    cv0 = 'CNNforELMo_0_score0.7570128353006951'
    cv1 = 'CNNforELMo_1_score0.7642858148797723'
    cv2 = 'CNNforELMo_2_score0.7694427833594161'
    cv3 = 'CNNforELMo_3_score0.7749008613135695'
    cv4 = 'CNNforELMo_4_score0.7658161433354106'

    pred_probs = {}
    pred_probs, test_prob = val_fold(cv0, dataset, pred_probs)
    x_test = np.zeros(test_prob.shape)
    x_test += test_prob
    dataset.change_fold(1)

    pred_probs, test_prob = val_fold(cv1, dataset, pred_probs)
    x_test += test_prob
    dataset.change_fold(2)

    pred_probs, test_prob = val_fold(cv2, dataset, pred_probs)
    x_test += test_prob
    dataset.change_fold(3)

    pred_probs, test_prob = val_fold(cv3, dataset, pred_probs)
    x_test += test_prob
    dataset.change_fold(4)

    pred_probs, test_prob = val_fold(cv4, dataset, pred_probs)
    x_test += test_prob
    # pred_probs.sort(key=lambda x: x[0])
    # iddd, xxx = zip(*pred_probs)
    train_id = pd.read_csv('data/multilabel.csv')['id']
    xxx = []
    for i in train_id:
        xxx.append(pred_probs[i])
    xxx = np.array(xxx)
    print(xxx.shape)
    # xxx = xxx.reshape((-1,))

    # predicts.sort(key=lambda x: x[0])
    # iddd, yyy = zip(*predicts)
    # yyy = np.array(yyy)
    # print(yyy.shape)
    # yyy = yyy.reshape((-1,))

    # temp = pd.read_csv('result/prob_rcnnoaug_cv.csv')
    # temp = temp.drop('id', axis=1)
    x_test /= 5
    joblib.dump((xxx, x_test), 'elmo_rcnn.pk')


if __name__ == "__main__":
    fire.Fire()