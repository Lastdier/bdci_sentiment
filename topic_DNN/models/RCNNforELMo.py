import torch as t
import numpy as np
from torch import nn
import json


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)


class RCNNforELMo(nn.Module): 
    def __init__(self, opt ):
        super(RCNNforELMo, self).__init__()
        self.model_name = 'RCNNforELMo'
        self.opt=opt

        kernel_size = opt.kernel_size

        self.content_lstm =nn.LSTM(  input_size = 1024,
                            hidden_size = opt.hidden_size,
                            num_layers = opt.num_layers,
                            bias = True,
                            batch_first = False,
                            # dropout = 0.5,
                            bidirectional = True
                            )

        self.content_conv = nn.Sequential(
            nn.Conv1d(in_channels = opt.hidden_size*2 + 1024,
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
            nn.Linear(opt.kmax_pooling*(opt.content_dim),opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )

    def forward(self, content):
        content_out = self.content_lstm(content.permute(1,0,2))[0].permute(1,2,0)
        content_em = (content).permute(0,2,1)
        content_out = t.cat((content_out,content_em),dim=1)

        content_conv_out = kmax_pooling(self.content_conv(content_out),2,self.opt.kmax_pooling)

        reshaped = content_conv_out.view(content_conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits


if __name__ == '__main__':
    pass
