from .BasicModule import BasicModule
import torch as torch
import numpy as np
from torch import nn
import json

class Multitask(BasicModule): 
    def __init__(self, opt ):
        super(Multitask, self).__init__()
        self.model_name = 'Multitask'
        self.opt=opt

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
        
        self.fc1 = nn.Linear(2*opt.hidden_size, 2*opt.hidden_size)
        # self.fc2 = nn.Linear(addition_feature_size, opt.hidden_size)
        self.act1 = nn.Tanh()
        self.act2 = nn.ReLU(True)
        self.fc3 = nn.Linear(2*opt.hidden_size, opt.linear_hidden_size)
        self.mlps = [nn.Linear(opt.linear_hidden_size, 3) for _ in range(10)]

    def forward(self, content, topic):
        embedded = self.encoder(content)
        lstm_out = self.content_lstm(embedded.permute(1,0,2))[0].permute(1,0,2)
        # (B, L, H)
        lstm_out = self.fc1(lstm_out)
        lstm_out = self.act1(lstm_out)
        lstm_out = self.act2(self.fc3(lstm_out))
        out = self.mlps[topic[0]](lstm_out)
        return out
    
    def load_embedding(self, myembedding):
        path = self.opt.type_ + '2index.json'
        f = open(path, 'r')
        word2index = json.load(f)
        f.close()
        
        weight = np.random.uniform(-0.1,0.1,[self.opt.vocab_size, len(myembedding)])
        weight = np.concatenate([weight, np.zeros((1,len(myembedding)))], 0)
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