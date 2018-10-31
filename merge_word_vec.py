import json
import numpy as np
from sklearn.externals import joblib
import fire


def merge_embedding(type_, word2vec, glove):
    path = type_ + '2index.json'
    f = open(path, 'r')
    word2index = json.load(f)
    f.close()

    pretrained = MyEmbeddings('sgns.weibo.bigram-char') 
    myword2vec = MyEmbeddings(word2vec)
    word_dict = {}
    for line in pretrained:
        pair = line.split(' ')
        if word2index.get(pair[0]) is not None:
            word_dict[word2index[pair[0]]] = [float(i) for i in pair[1:]]
    for line in myword2vec:
        pair = line.split(' ')
        if word2index.get(pair[0]) is None:
            continue
        if word_dict.get(word2index[pair[0]]) is None:
            temp = np.random.uniform(-0.1,0.1,[1, len(pretrained)]).reshape((len(pretrained),))
            temp = temp.tolist()
            word_dict[word2index[pair[0]]] = temp
        if pair[1] == '':
            continue
        word_dict[word2index[pair[0]]] += [float(i) for i in pair[1:]]
    myglove = MyEmbeddings(glove)
    for line in myglove:
        pair = line.split(' ')
        if word2index.get(pair[0]) is None:
            continue
        word_dict[word2index[pair[0]]] += [float(i) for i in pair[1:]]
    
    #count = 0
    weight_pad = np.zeros((1,len(pretrained)+200))
    weight = np.random.uniform(-0.1,0.1,[len(word2index)-1, len(pretrained)+200])
    weight = np.concatenate([weight_pad, weight], 0)
    for w in word2index:
        if word_dict.get(w) is not None:
            weight[word2index[w]] = word_dict[w]
    print(weight.dtype)
    print(weight.shape)
    joblib.dump(weight, 'mixed_'+type_+'_500.pk')
    

class MyEmbeddings(object):
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for line in open(self.path, 'r'):
            yield line.strip()

    def __len__(self):
        length = 0
        with open(self.path, 'r') as f:
            length = f.readline().split()[1]
        return int(length)


if __name__ == '__main__':
    fire.Fire()
