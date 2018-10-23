import torch
import fire
from data.dataset import My_dataset
import os
from torch.utils import data
import pandas as pd
import numpy as np
import models
from config import opt
import tqdm
from sklearn.metrics import f1_score


def main(**kwargs):
    opt.parse(kwargs,print_=True)
    model = getattr(models,opt.model)(opt).cuda()
    print(model)
    lr,lr2=opt.lr,opt.lr2
    loss_function = getattr(models,opt.loss)()  

    dataset = My_dataset(opt.seq_len, augment=opt.augument)
    dataloader = data.DataLoader(dataset,
                    batch_size = opt.batch_size,
                    shuffle = True,
                    num_workers = 4,
                    pin_memory = True
                    )

    optimizer = model.get_optimizer(lr,lr2)

    pred_probs = []
    batch_count = 0
    not_increase_count = 0
    pre_f1 = 0.
    f1 = 0.
    for epoch in range(opt.max_epoch):
        
        for ii,(content,label,sen_id) in enumerate(dataloader):
            content,label = content.cuda(),label.cuda()
            
            optimizer.zero_grad()
            score = model(content)

            #proba = score.detach().cpu().numpy()
            # lll = label.cpu().numpy()
            # mul = []
            # for iiiii in range(len(lll)):
            #     ccc = 0
            #     for jjjj in range(len(lll[0])):
            #         if lll[iiiii][jjjj] == 1:
            #             ccc += 1
            #     if ccc > 1:
            #         mul.append(iiiii)
            # print(proba[mul])
            # print(lll[mul])

            predict = score.detach().cpu().numpy()
            predict_ind = np.zeros((predict.shape[0], 10), dtype=np.int32)
            for i in range(predict.shape[0]):
                ttt = predict[i]
                tttt = [ttt>0.]
                predict_ind[i][tttt] = 1
            #print(score.detach().cpu().numpy())
            #print(label.cpu().numpy())
            loss = loss_function(score, label)
            loss.backward()
            optimizer.step()
            
            f1 += f1_score(label.cpu().numpy(), predict_ind, average='macro')
            if batch_count%opt.plot_every ==opt.plot_every-1:
                
                # compute average f1 score
                f1 = f1 / opt.plot_every

                #eval()
                print('average f1: %f' % f1)
                if f1 < pre_f1:
                    not_increase_count += 1
                else:
                    not_increase_count = 0
                
                if not_increase_count > 3:
                    if lr <= opt.min_lr:
                        break
                    lr *= opt.lr_decay
                    lr2 *= 0.8
                    optimizer = model.get_optimizer(lr,lr2)


                pre_f1 = f1
                f1 = 0.
            if lr <= opt.min_lr:
                break
            batch_count += 1
    torch.save(model.cpu().state_dict(), 'cnn.pt')




if __name__ == '__main__':
    fire.Fire()