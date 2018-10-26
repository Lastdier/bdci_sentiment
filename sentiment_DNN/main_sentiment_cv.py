import torch
import fire
from data.dataset import My_sentiment_dataset
import os
from torch.utils import data
import pandas as pd
import numpy as np
import models
from config import opt
import tqdm
from sklearn.metrics import f1_score


def val(model,dataset):
    dataset.change2val()

    loss_function = getattr(models,opt.loss)() 

    dataloader = data.DataLoader(dataset,
                    batch_size = opt.batch_size,
                    shuffle = False,
                    num_workers = opt.num_workers,
                    pin_memory = True
                    )
    
    loss = 0.
    f1_label = []
    f1_predict = []
    with torch.no_grad():
        for i, (content, characters, label, topic) in tqdm.tqdm(enumerate(dataloader)):
            if opt.type_ == 'word':
                content,label = content.cuda(),label.cuda()
            elif opt.type_ == 'char':
                content,label = characters.cuda(),label.cuda()
            score = model(content, topic)

            loss += loss_function(score, label.squeeze(1))
            predict = score.data.topk(1,dim=1)[1].cpu().tolist()
            
            f1_label += label.cpu().tolist()
            f1_predict += predict


            
    del score

    ave_f1 = f1_score(f1_label, f1_predict, average='macro')

    dataset.change2train()

    wrong_label = None
    return loss.item(), ave_f1, wrong_label


def main(**kwargs):
    opt.parse(kwargs,print_=True)
    model = getattr(models,opt.model)(opt).cuda()
    print(model)
    lr,lr2=opt.lr,opt.lr2
    loss_function = getattr(models,opt.loss)()  

    dataset = My_sentiment_dataset(opt.seq_len, cv=True, augment=opt.augument)
    print(len(dataset))
    dataloader = data.DataLoader(dataset,
                    batch_size = opt.batch_size,
                    shuffle = True,
                    num_workers = 4,
                    pin_memory = True
                    )

    optimizer = model.get_optimizer(lr,lr2,weight_decay=opt.weight_decay)
    best_score = 0

    pred_probs = []
    for fold in range(5):
        if fold > 0:
            dataset.change_fold(fold)
            dataloader = data.DataLoader(dataset,
                        batch_size = opt.batch_size,
                        shuffle = True,
                        num_workers = 4,
                        pin_memory = True
                        )
            del model
            del optimizer
            best_score = 0
            model = getattr(models,opt.model)(opt).cuda()
            print(model)
            lr,lr2=opt.lr,opt.lr2
            optimizer = model.get_optimizer(opt.lr,opt.lr2,weight_decay=opt.weight_decay)
        batch_count = 0
        notimproved_count = 0
        f1_label = []
        f1_predict = []
        for epoch in range(opt.max_epoch):
            for i,(content, characters, label, topic) in tqdm.tqdm(enumerate(dataloader)):
                if opt.type_ == 'word':
                    content,label = content.cuda(),label.cuda()
                elif opt.type_ == 'char':
                    content,label = characters.cuda(),label.cuda()
                optimizer.zero_grad()
                score = model(content, topic)
                predict = score.data.topk(1,dim=1)[1].cpu().tolist()
                loss = loss_function(score, label.squeeze(1))
                
                loss.backward()
                optimizer.step()
                
                f1_label += label.cpu().tolist()
                f1_predict += predict
                
                if batch_count%opt.plot_every ==opt.plot_every-1:
                    
                    # compute average f1 score
                    f1 = f1_score(f1_label, f1_predict, average='macro')
                    f1_label = []
                    f1_predict = []
                    #eval()
                    # print('train average f1: %f' % f1)
                    #k = torch.randperm(label.size(0))[0]

                if batch_count%opt.decay_every == opt.decay_every-1:                       
                    del loss
                    val_loss, val_f1, wrong_label= val(model,dataset)
                    if val_f1 <= best_score:
                        notimproved_count += 1
                        if notimproved_count == opt.early_stoping or lr <= opt.min_lr:
                            break
                        state = torch.load(opt.model+'_'+str(fold)+'_score' +str(best_score)+'.pt')
                        model.load_state_dict(state)
                        model.cuda()
                        lr = lr * opt.lr_decay
                        lr2= 2e-4 if lr2==0 else  lr2*0.8
                        optimizer = model.get_optimizer(lr,lr2)  
                    if val_f1 > best_score:
                        notimproved_count = 0
                        best_score = val_f1
                        torch.save(model.cpu().state_dict(), opt.model+'_'+str(fold)+'_score'+str(best_score)+'.pt')
                        model.cuda()               

                batch_count += 1
            if notimproved_count == opt.early_stoping or lr <= opt.min_lr:
                break

if __name__ == '__main__':
    fire.Fire()