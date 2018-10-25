import torch as torch
import numpy as np
from torch import nn
import json


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)

class LSTMforELMo(nn.Module): 
    def __init__(self, opt ):
        super(LSTMforELMo, self).__init__()
        self.model_name = 'LSTMforELMo'
        self.opt=opt
        
        self.content_lstm =nn.LSTM(input_size = 1024,
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
        content_out = self.content_lstm(content.permute(1,0,2))[0].permute(1,2,0)

        content_conv_out = kmax_pooling((content_out),2,self.opt.kmax_pooling)

        reshaped = content_conv_out.view(content_conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits
