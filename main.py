from torch.utils.data import DataLoader 
from learner import Learner
from loss import * 
from dataset import AnomalyDataset, NormalDataset

import argparse
import os 
from sklearn import metrics 
import numpy as np


def train(model, epoch, optimizer, scheduler, criterion, dataloaders):
    normal_dl, anomal_dl = dataloaders

    print("\n Epoch: %d" % epoch) 
    model.train()
    train_loss = 0.
    correct = 0
    total = 0

    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_dl, anomal_dl)):
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim =1) # [32,2048]
        batch_size = inputs.shape[0]
        inputs = input.view(-1, inputs.size(-1).to(device))
        outputs = model(inputs)
        loss = criterion(outputs, batch_size)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    print('loss : {}'.format(train_loss/ len(normal_dl)))
    scheduler.step()

def test_abnormal(epoch, model, dataloaders, device):
    normal_dl, anomal_dl = dataloaders
    model.eval()
    auc = 0

    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(normal_dl, anomal_dl)):
            inputs, gts, frames = data
            inputs = inputs.view(-1, inputs.size(-1)).to(device)
            score = model(inputs)
            score =score.cpu().detach().numpy()
            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0]//16, 33))
            
            for j in range(32):
                score_list[int(step[j]) * 16: (int(step[j+1])) * 16] = score[j] 
            
            gt_list = np.zeros(frames[0])
            for k in range(len(gts)//2):
                s = gts[k*2]
                e = min(gts[k*2 +1], frames)
                gt_list[s-1:e] =1
            
            inputs2, gts2, frames2 = data2 
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(device) 
            score2 = model(inputs2)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, frames[0]//16, 33))
            for kk in range(32):
                score_list[int(step2[kk])*16 : (int(step2[kk+1]))* 16] = score2[kk]

            gt_list2 = np.zeros(frames2[0])
            score_list3 = np.concatenate((score_list, score_list2), axis= 0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)

            fpr, tpr, thresholds = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            auc += metrics.auc(fpr, tpr)
        print('auc = ', auc/140) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Difine Hyperpaarms')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=75, type=int)
    normal_train_dataset = NormalDataset(is_train=1)
    normal_test_dataset = NormalDataset(is_train=0)

    anomaly_train_dataset = AnomalyDataset(is_train=1)
    anomaly_test_dataset = AnomalyDataset(is_train=0) 

    normal_train_dataloader = DataLoader(normal_train_dataset, batch_size= parser.batch_size)  
    normal_test_dataloader = DataLoader(normal_test_dataset, batch_size=parser.batch_size)

    anomaly_train_dataloader = DataLoader(anomaly_train_dataset, batch_size= parser.batch_size)
    anomaly_test_dataloader = DataLoader(anomaly_test_dataset, batch_size = parser.batch_size)  

    device = 'cpu'
    if torch.cuda.is_available() or torch.backends.mps.is_available() : 
        if torch.cuda.is_available() : 
            device = 'cuda'
        else :
            device = 'mps'
    

    model = Learner(input_dim=2048, drop_p=0.0).to(device) 
    optimizer = torch.optim.Adagrad(model.parameters(), lr = 1e-3, weight_decay= 0.0010004)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= [25,50])
    criterion = MIL


    for epoch in range(0,parser.epochs):
        train(model,epoch, optimizer, scheduler, criterion, [normal_train_dataloader, anomaly_train_dataloader])
        test_abnormal(epoch,model,[normal_test_dataloader, anomaly_test_dataloader])
        