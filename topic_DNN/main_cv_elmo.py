import torch
import fire
from data.datasetWithELMo import My_dataset
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
        for i,(content, label, _) in tqdm.tqdm(enumerate(dataloader)):
            content = content.cuda()
            label = label.cuda()
            score = model(content)

            loss += loss_function(score, label)
            predict = score.detach().cpu().numpy()
            predict_ind = np.zeros((predict.shape[0], 10), dtype=np.int32)
            for i in range(predict.shape[0]):
                ttt = predict[i]
                tttt = [ttt>0.]
                predict_ind[i][tttt] = 1
            f1_predict.append(predict_ind)
            f1_label.append(label.cpu().numpy())

            
    del score

    ave_f1 = f1_score(np.concatenate(f1_label, axis=0), np.concatenate(f1_predict, axis=0), average='macro')

    dataset.change2train()

    wrong_label = None
    return loss.item(), ave_f1, wrong_label


def main(**kwargs):
    opt.parse(kwargs,print_=True)
    model = getattr(models,opt.model)(opt).cuda()
    print(model)
    lr,lr2=opt.lr,opt.lr2
    loss_function = getattr(models,opt.loss)()  

    dataset = My_dataset(cv=True)
    dataloader = data.DataLoader(dataset,
                    batch_size = opt.batch_size,
                    shuffle = True,
                    num_workers = 4,
                    pin_memory = True
                    )

    optimizer = torch.optim.Adam(params=model.parameters(), weight_decay = opt.weight_decay,lr=lr)
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
            optimizer = torch.optim.Adam(params=model.parameters(), weight_decay = opt.weight_decay,lr=lr)
        batch_count = 0
        notimproved_count = 0
        f1_label = []
        f1_predict = []
        for epoch in range(opt.max_epoch):
            for i,(content, label, sen_id) in tqdm.tqdm(enumerate(dataloader)):
                content = content.cuda()
                label = label.cuda()
                score = model(content)
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
                
                f1_predict.append(predict_ind)
                f1_label.append(label.cpu().numpy())
                if batch_count%opt.plot_every ==opt.plot_every-1:
                    
                    # compute average f1 score
                    f1 = f1_score(np.concatenate(f1_label, axis=0), np.concatenate(f1_predict, axis=0), average='macro')
                    f1_label = []
                    f1_predict = []

                    #eval()
                    print('train average f1: %f' % f1)
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
                        optimizer = torch.optim.Adam(params=model.parameters(), weight_decay = opt.weight_decay,lr=lr)  
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