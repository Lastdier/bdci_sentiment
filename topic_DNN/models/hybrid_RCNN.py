from .BasicModule import BasicModule
import torch as t
import numpy as np
from torch import nn
import json


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)


class hybrid_RCNN(BasicModule): 
    def __init__(self, opt ):
        super(hybrid_RCNN, self).__init__()
        self.model_name = 'hybrid_RCNN'
        self.opt=opt

        self.vocab_size = opt.word_size + opt.char_size
        self.word_len = opt.word_size # 字母下标为 len(word2index) + 原下标
        self.word2index = opt.word2index
        self.char2index = opt.char2index

        kernel_size = opt.kernel_size
        self.encoder = nn.Embedding(self.vocab_size,opt.embedding_dim, padding_idx=0)

        if opt.embedding_path:
            self.encoder.from_pretrained(self.load_embedding(MyEmbeddings(opt.embedding_path)))

        self.content_lstm =nn.LSTM(  input_size = opt.embedding_dim,
                            hidden_size = opt.hidden_size,
                            num_layers = opt.num_layers,
                            bias = True,
                            batch_first = False,
                            # dropout = 0.5,
                            bidirectional = True
                            )

        self.char_lstm =nn.LSTM(  input_size = opt.embedding_dim,
                            hidden_size = opt.hidden_size,
                            num_layers = opt.num_layers,
                            bias = True,
                            batch_first = False,
                            # dropout = 0.5,
                            bidirectional = True
                            )

        self.content_conv = nn.Sequential(
            nn.Conv1d(in_channels = opt.hidden_size*2 + opt.embedding_dim,
                      out_channels = opt.content_dim,
                      kernel_size =  kernel_size),
            nn.BatchNorm1d(opt.content_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels = opt.content_dim,
                      out_channels = opt.content_dim,
                      kernel_size =  kernel_size),
            nn.BatchNorm1d(opt.content_dim),
            nn.ReLU(inplace=True),
        )
        self.char_conv = nn.Sequential(
            nn.Conv1d(in_channels = opt.hidden_size*2 + opt.embedding_dim,
                      out_channels = opt.content_dim,
                      kernel_size =  kernel_size),
            nn.BatchNorm1d(opt.content_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels = opt.content_dim,
                      out_channels = opt.content_dim,
                      kernel_size =  kernel_size),
            nn.BatchNorm1d(opt.content_dim),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(opt.kmax_pooling*(opt.content_dim*2),opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )

    def forward(self, content, char):
        content = self.encoder(content)
        char = self.encoder(t.add(char, self.word_len))

        content_out = self.content_lstm(content.permute(1,0,2))[0].permute(1,2,0)
        content_em = content.permute(0,2,1)
        content_out = t.cat((content_out,content_em),dim=1)
        char_out = self.char_lstm(char.permute(1,0,2))[0].permute(1,2,0)
        char_em = char.permute(0,2,1)
        char_out = t.cat((char_out,char_em),dim=1)

        content_conv_out = kmax_pooling(self.content_conv(content_out),2,self.opt.kmax_pooling)
        char_conv_out = kmax_pooling(self.char_conv(char_out),2,self.opt.kmax_pooling)

        conv_out = t.cat((char_conv_out, content_conv_out), dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits
    
    def load_embedding(self, myembedding):
        weight = np.random.uniform(-0.1,0.1,[self.vocab_size, len(myembedding)])
        
        weight = np.concatenate([weight, np.zeros((1,len(myembedding)))], 0)
        count = 0
        for line in myembedding:
            pair = line.split(' ')
            if self.word2index.get(pair[0]) is not None:
                count += 1
                weight[self.word2index[pair[0]]] = [float(i) for i in pair[1:]]
            elif self.char2index.get(pair[0]) is not None:
                count += 1
                weight[self.char2index[pair[0]] + self.word_len] = [float(i) for i in pair[1:]]
        print(count)
        weight = t.tensor(weight, dtype=t.float32)
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


if __name__ == '__main__':
    pass
