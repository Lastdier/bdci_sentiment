import pandas as pd
import numpy as np
import jieba
import json
import random
import math


train = pd.read_csv('data/train.csv')
jieba.load_userdict('userdict.txt')
SUBJECT_MASK = {'价格': 0, '配置': 1, '操控': 2, '舒适性': 3, '油耗': 4, '动力': 5, '内饰': 6, '安全性': 7, '空间': 8, '外观': 9}
SUBJECT_LIST = ['价格', '配置', '操控', '舒适性', '油耗', '动力', '内饰', '安全性', '空间', '外观']
def subject_distribution():
    subject = train['subject']
    subject_stat = {}
    for i in subject:
        if subject_stat.get(i) is None:
            subject_stat[i] = 0
        else:
            subject_stat[i] += 1

    print(subject_stat)


def sentiment_distribution():
    for i in subject_stat:
        print(i.decode('utf-8'), subject_stat[i])
    sentiment = train['sentiment_value']
    sentiment_stat = {}
    for i in sentiment:
        if sentiment_stat.get(i) is None:
            sentiment_stat[i] = 0
        else:
            sentiment_stat[i] += 1
    print(sentiment_stat)


def sentiment_word_distribution():
    sentiment = train['sentiment_value']
    sentiment_word = train['sentiment_word']
    subject = train['subject']
    dis = {'价格': {0:[], 1:[], -1:[]}, '配置':  {0:[], 1:[], -1:[]}, '操控':  {0:[], 1:[], -1:[]}, '舒适性':  {0:[], 1:[], -1:[]}, '油耗':  {0:[], 1:[], -1:[]}, '动力':  {0:[], 1:[], -1:[]}, '内饰':  {0:[], 1:[], -1:[]}, '安全性':  {0:[], 1:[], -1:[]}, '空间':  {0:[], 1:[], -1:[]}, '外观':  {0:[], 1:[], -1:[]}}
    for i in range(len(sentiment)):
        if type(sentiment_word[i]) == type('str'):
            if len(sentiment_word[i]) > 3:
                word_temp = jieba.lcut(sentiment_word[i], cut_all=False)
            else:
                word_temp = sentiment_word[i].split(' ')
            for w in word_temp:
                if w in [' ', ',', '', '/', '，', '。']:
                    continue
                dis[subject[i]][sentiment[i]].append(w)
    # print(dis) 每一个主题下，情感词在每一情感下出现的概率
    bdc = {'价格': {}, '配置': {}, '操控': {}, '舒适性': {}, '油耗': {}, '动力': {}, '内饰': {}, '安全性': {}, '空间': {}, '外观': {}}
    for label in dis:
        temp = {}
        for sentiment in dis[label]:
            total = len(dis[label][sentiment])
            for word in dis[label][sentiment]:
                if temp.get(word) is None:
                    temp[word] = [0.,0.,0.]
                temp[word][sentiment] += 1. / total
        for word in temp:
            word_bdc = 1
            sss = sum(temp[word])
            for cl_count in temp[word]:
                if cl_count == 0:
                    continue
                # word_bdc += math.log(cl_count*1.0/sss)*(cl_count*1.0/sss)/math.log(3)
                word_bdc += math.log(cl_count/sss)*(cl_count/sss)/math.log(3)
            bdc[label][word] = word_bdc
    outfile = open('bdcbytopics.json', 'w')
    outfile.write(json.dumps(bdc))
    outfile.close()
    #print(bdc)
    #print(json.dumps(dis))



def tokenization(min_count=1):
    vol_dict = {}
    for i in train['content']:
        sentence = i.strip()
        sentence = sentence.replace('，', ',')
        sentence = sentence.replace('？', '?')
        sentence = sentence.replace('！', '!')
        sentence = jieba.cut(sentence, cut_all=False)
        for j in sentence:
            if vol_dict.get(j) is None:
                vol_dict[j] = 1
            else:
                vol_dict[j] += 1
    # load test
    test = pd.read_csv('data/test_public.csv')
    for i in test['content']:
        sentence = i.strip()
        sentence = sentence.replace(' ', '')
        sentence = sentence.replace('，', ',')
        sentence = sentence.replace('？', '?')
        sentence = sentence.replace('！', '!')
        sentence = jieba.cut(sentence, cut_all=False)
        for j in sentence:
            if vol_dict.get(j) is None:
                vol_dict[j] = 1
            else:
                vol_dict[j] += 1
    word2index = {'<pad>': 0,
                    '<oov>': 1}
    pointer = 2
    for i in vol_dict:
        # filter
        if vol_dict[i] > min_count-1:
            word2index[i] = pointer
            pointer += 1
    
    with open('topic_DNN/word2index.json', 'w', encoding='utf-8') as outfile:
        json.dump(word2index, outfile)


def character_tokenizer(min_count=1):
    vol_dict = {}
    for i in train['content']:
        sentence = i.strip()
        sentence = sentence.replace(' ', '')
        sentence = sentence.replace('，', ',')
        sentence = sentence.replace('？', '?')
        sentence = sentence.replace('！', '!')
        for j in sentence:
            if vol_dict.get(j) is None:
                vol_dict[j] = 1
            else:
                vol_dict[j] += 1
    # load test
    test = pd.read_csv('data/test_public.csv')
    for i in test['content']:
        sentence = i.strip()
        sentence = sentence.replace(' ', '')
        sentence = sentence.replace('，', ',')
        sentence = sentence.replace('？', '?')
        sentence = sentence.replace('！', '!')
        for j in i:
            if vol_dict.get(j) is None:
                vol_dict[j] = 1
            else:
                vol_dict[j] += 1
    char2index = {'<pad>': 0,
                    '<oov>': 1}
    pointer = 2
    for i in vol_dict:
        # filter
        if vol_dict[i] > min_count-1:
            char2index[i] = pointer
            pointer += 1
    
    with open('topic_DNN/char2index.json', 'w', encoding='utf-8') as outfile:
        json.dump(char2index, outfile)


def partition(vector, left, right, pivotIndex):
    pivotValue = vector[pivotIndex]
    vector[pivotIndex], vector[right] = vector[right], vector[pivotIndex]  # Move pivot to end
    storeIndex = left
    for i in range(left, right):
        if vector[i] < pivotValue:
            vector[storeIndex], vector[i] = vector[i], vector[storeIndex]
            storeIndex += 1
    vector[right], vector[storeIndex] = vector[storeIndex], vector[right]  # Move pivot to its final place
    return storeIndex
 
def _select(vector, left, right, k):
    "Returns the k-th smallest, (k >= 0), element of vector within vector[left:right+1] inclusive."
    while True:
        pivotIndex = random.randint(left, right)     # select pivotIndex between left and right
        pivotNewIndex = partition(vector, left, right, pivotIndex)
        pivotDist = pivotNewIndex - left
        if pivotDist == k:
            return vector[pivotNewIndex]
        elif k < pivotDist:
            right = pivotNewIndex - 1
        else:
            k -= pivotDist + 1
            left = pivotNewIndex + 1
 
def select(vector, k, left=None, right=None):
    """\
    Returns the k-th smallest, (k >= 0), element of vector within vector[left:right+1].
    left, right default to (0, len(vector) - 1) if omitted
    """
    if left is None:
        left = 0
    lv1 = len(vector) - 1
    if right is None:
        right = lv1
    assert vector and k >= 0, "Either null vector or k < 0 "
    assert 0 <= left <= lv1, "left is out of range"
    assert left <= right <= lv1, "right is out of range"
    return _select(vector, left, right, k)
 
def get_95_shortest():
    lens = []
    cl = len(train["content"])

    for w in train["content"]:
        sentence = jieba.lcut(w.strip(), cut_all=False)
        temp=''
        for i in sentence:
            if i != ' ':
                temp += i
        lens.append(len(temp))
    print(np.array(lens).mean())
    print(select(lens, int(cl*0.95)))


if __name__ == '__main__':
    character_tokenizer()
