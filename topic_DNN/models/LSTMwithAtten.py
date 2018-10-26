from .BasicModule import BasicModule
import torch as torch
import numpy as np
from torch import nn
import json


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # self.projection = nn.Sequential(
        #     nn.Linear(hidden_dim, 64),
        #     nn.ReLU(True),  # 激活函数
        #     nn.Linear(64, 1)
        # )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),  # 激活函数
        )
        self.uw_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),  # 激活函数
        )

    def forward(self, encoder_outputs):
        # (B, L, H)
        u_it = self.projection(encoder_outputs)
        # (B, L, H)
        # TODO: 如何正确取出代表句子表示的最后一步隐含层输出，u_w实现是否正确
        forward = encoder_outputs[:,-1,:hidden_dim/2]
        backward = encoder_outputs[:,1,hidden_dim/2:]
        represent = torch.cat((forward, backward), dim=-1)
        u_w = self.uw_projection(represent).unsqueeze(1).permute(0, 2, 1)   #取出encoder最后的一步的输出
        # (B, H, 1)
        mul = torch.bmm(u_it, u_w)
        # (B, L, 1)
        weights = torch.nn.functional.softmax(mul)
        # (B, H, L) * (B, L, 1) -> (B, H)
        outputs = torch.bmm(encoder_outputs.permute(0, 2, 1), weights).squeeze()
        return outputs, weights.squeeze()


class LSTMwithAtten(BasicModule): 
    def __init__(self, opt ):
        super(LSTMwithAtten, self).__init__()
        self.model_name = 'LSTMwithAtten'
        self.opt=opt

        self.encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim, padding_idx=0)

        if opt.embedding_path:
            self.encoder.from_pretrained(self.load_embedding(MyEmbeddings(opt.embedding_path)))
        
        self.content_lstm =nn.LSTM(input_size = opt.embedding_dim,
                            hidden_size = opt.hidden_size,
                            num_layers = 1,
                            bias = True,
                            batch_first = False,
                            bidirectional = True
                            )

        self.attention = SelfAttention(opt.hidden_size*2)

        self.fc = nn.Sequential(
            nn.Linear(opt.hidden_size*2,opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )

    def forward(self, content):
        content = self.encoder(content)
        if self.opt.static:
            content=content.detach()

        content_out = self.content_lstm(content.permute(1,0,2))[0].permute(1,0,2)
        # [B, L, H]
        out, weights = self.attention(content_out)

        # [opt.hidden_size * 2, opt.hidden_size * 2]

        reshaped = out.view(out.size(0), -1)
        logits = self.fc((reshaped))
        return logits, weights
    
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
