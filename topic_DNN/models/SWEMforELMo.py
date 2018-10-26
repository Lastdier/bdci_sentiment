import torch as t
import numpy as np
import json
from torch import nn


class SWEMforELMo(nn.Module): 
    def __init__(self, opt):
        super(SWEMforELMo, self).__init__()
        self.model_name = 'SWEMforELMo'
        self.opt=opt

        self.seq_len = 100
        
        if opt.swem_type == 'max':
            self.pooling = nn.MaxPool1d(kernel_size=self.seq_len)
        elif opt.swem_type == 'ave':
            self.pooling = nn.AvgPool1d(kernel_size=self.seq_len)

        self.fc = nn.Sequential(
            nn.Linear(1024, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000,opt.num_classes)
        )
        

    def forward(self, content):
        # (B, L, C)
        content = self.pooling(content.permute(0, 2, 1))
        # (B, C, 1)
        logits = self.fc(content.squeeze())
        return logits

 
if __name__ == '__main__':
    pass
