from .BasicModule import BasicModule
import torch as t
import numpy as np
import json
from torch import nn

kernel_sizes =  [1,2,3,4]
class CNNforELMo(nn.Module): 
    def __init__(self, opt ):
        super(CNNforELMo, self).__init__()
        self.model_name = 'CNNforELMo'
        self.opt=opt

        content_convs = [ nn.Sequential(
                                nn.Conv1d(in_channels = 1024,
                                        out_channels = opt.content_dim,
                                        kernel_size = kernel_size),
                                nn.BatchNorm1d(opt.content_dim),
                                nn.ReLU(inplace=True),
                                # nn.MaxPool1d(kernel_size = (100 - kernel_size + 1))
                                nn.Conv1d(in_channels = opt.content_dim,
                                        out_channels = opt.content_dim,
                                        kernel_size = kernel_size),
                                nn.BatchNorm1d(opt.content_dim),
                                nn.ReLU(inplace=True),
                                nn.MaxPool1d(kernel_size = (100 - kernel_size*2 + 2))
                            )
            for kernel_size in kernel_sizes ]

        self.content_convs = nn.ModuleList(content_convs)

        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes)*(opt.content_dim), 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000,opt.num_classes)
        )
        
        

    def forward(self, content):
        content_out = [content_conv(content.permute(0,2,1)) for content_conv in self.content_convs]
        conv_out = t.cat((content_out),dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits

 
if __name__ == '__main__':
    pass
