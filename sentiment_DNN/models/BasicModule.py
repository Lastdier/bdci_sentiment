#coding:utf8
import torch
import time

class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))   

    def get_optimizer(self,lr1,lr2=0,weight_decay = 0):
        ignored_params = list(map(id, self.encoder.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                        self.parameters())
        if lr2 is None: lr2 = lr1*0.5 
        optimizer = torch.optim.Adam([
                dict(params=base_params,weight_decay = weight_decay,lr=lr1),
                {'params': self.encoder.parameters(), 'lr': lr2}
            ])
        return optimizer