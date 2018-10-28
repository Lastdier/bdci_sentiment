from .BasicModule import BasicModule
import torch as torch
import numpy as np
from torch import nn
import json


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)

class LSTMText(BasicModule): 
    def __init__(self, opt ):
        super(LSTMText, self).__init__()
        self.model_name = 'LSTMText'
        self.opt=opt

        kernel_size = opt.kernel_size
        self.encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim, padding_idx=0)

        if opt.embedding_path:
            self.encoder.from_pretrained(self.load_embedding(MyEmbeddings(opt.embedding_path)))
        
        self.content_lstm =nn.LSTM(input_size = opt.embedding_dim,
                            hidden_size = opt.hidden_size,
                            num_layers = opt.num_layers,
                            bias = True,
                            batch_first = False,
                            bidirectional = True
                            )

        self.fc = nn.Sequential(
            nn.Linear(opt.kmax_pooling*(opt.hidden_size*2),opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )

    def forward(self, content):
        content = self.encoder(content)
        if self.opt.static:
            content=content.detach()

        content_out = self.content_lstm(content.permute(1,0,2))[0].permute(1,2,0)

        content_conv_out = kmax_pooling((content_out),2,self.opt.kmax_pooling)

        reshaped = content_conv_out.view(content_conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits
    
    def load_embedding(self, myembedding):
        path = self.opt.type_ + '2index.json'
        f = open(path, 'r')
        word2index = json.load(f)
        f.close()
        
        weight_pad = np.zeros((1,len(myembedding)))
        weight = np.random.uniform(-0.1,0.1,[self.opt.vocab_size-1, len(myembedding)])
        weight = np.concatenate([weight_pad, weight], 0)
        for line in myembedding:
            pair = line.split(' ')
            if word2index.get(pair[0]) is not None:
                weight[word2index[pair[0]]] = [float(i) for i in pair[1:]]

        weight = torch.tensor(weight, dtype=torch.float32)
        print('pretrain wordvec loaded!')
        return weight

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
