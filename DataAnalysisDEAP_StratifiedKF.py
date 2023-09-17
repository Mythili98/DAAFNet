#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:01:50 2023

@author: desktop
"""

import scipy
import modelsubjectdep
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

class Dataset(Dataset):
    def __init__(self, data, info):
        #data loading
        self.x = data
        self.y = info
        self.n_samples = data.shape[0]


    def __getitem__(self,index):
        t1 = self.x[index]
        t2 = self.y[index]
        t1 = torch.tensor(t1)
        t1 = t1.unsqueeze(0)
        t2 = torch.tensor(t2)
        return (t1,t2)
    
    def __len__(self):
        return self.n_samples
    
data = scipy.io.loadmat('/home/desktop/Desktop/22104412_Docs/EEG-COGMusic/data_s01_collapsed.mat')

d = data['data']
l = data['labels']

label_onehot = []
count_labels = {'lv_la': 0,
                'lv_ha':0,
                'hv_la':0,
                'hv_ha':0}
labels = []
for i in range(l.shape[0]):
    label = l[i,:]
    if ((label == [0.,0.]).all()).any():
        count_labels['lv_la'] +=1
        label_onehot.append([0])
        labels.append([1.,0.,0.,0.])
    elif ((label == [0.,1.]).all()).any():
        count_labels['lv_ha'] +=1
        label_onehot.append([1])
        labels.append([0.,1.,0.,0.])
    elif ((label == [1.,0.]).all()).any():
        count_labels['hv_la'] +=1
        label_onehot.append([2])
        labels.append([0.,0.,1.,0.])
    elif ((label == [1.,1.]).all()).any():
        count_labels['hv_ha'] +=1
        label_onehot.append([3])
        labels.append([0.,0.,0.,1.])

label_onehot = np.array(label_onehot)
labels = np.array(labels)
print('Participant s01 emotion label counts', count_labels)


trainLoss={}
valLoss = {}
trainAcc = {}
valAcc = {}


torch.autograd.set_detect_anomaly(True)
l2_lambda = 0.001
skf = StratifiedKFold(n_splits=10)
StratifiedKFold(n_splits=2, random_state=None, shuffle=True)

for k,(train_indices,val_indices) in enumerate(skf.split(d, label_onehot)):
    tL = []
    vL = []
    tAV = []
    tAA = []
    vAV = []
    vAC = []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = modelsubjectdep.ModelSubDep()
    model = model.to(device)
    print('Fold :', k)
    data = d
    labels = l
    dataV = data[val_indices,:,:]
    labelsV = labels[val_indices,:]
    dataTr = data[train_indices,:,:]
    labelsTr = labels[train_indices,:]
    trainDS = Dataset(dataTr, labelsTr)
    trainDL = DataLoader(dataset = trainDS, batch_size = 40, shuffle = True)
    
    valDS = Dataset(dataV, labelsV)
    valDL = DataLoader(dataset = valDS, batch_size = 40, shuffle = True)
    
    num_epochs = 200
    batchsize = 40
    criterion = nn.BCELoss()
    optimizer_ft = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
    for epoch in range(num_epochs):
        with tqdm(trainDL, unit='batch') as tepoch:
            trainAccuracy_valence = 0
            trainAccuracy_arousal = 0
            totalLoss = 0
            #print("Epoch {} running".format(epoch))# know what is the epoch running
            model.train()
            for images,lab in tepoch:
                images = images.to(device)
                lab = lab.to(device)
                optimizer_ft.zero_grad()
                outputs = (model(images.float()))
                predindex = (outputs>=0.5).float()
                loss = criterion(outputs,lab.float())
                l2_reg = 0.0
                for param in model.parameters():
                    l2_reg += torch.norm(param, p=2)  # L2 norm of the parameter
                loss += l2_lambda * l2_reg
                loss.backward()
                totalLoss+=loss.item()
                optimizer_ft.step()
                y_valence = (predindex[:,0]==lab[:,0]).sum().item()
                batchAccuracy_valence = y_valence/batchsize
                trainAccuracy_valence += y_valence
                y_arousal = (predindex[:,1]==lab[:,1]).sum().item()
                batchAccuracy_arousal = y_arousal/batchsize
                trainAccuracy_arousal += y_arousal
                totaltrainAccuracybatch_valence = trainAccuracy_valence/(len(trainDL)*batchsize)
                totaltrainAccuracybatch_arousal = trainAccuracy_arousal/(len(trainDL)*batchsize)
                tepoch.set_postfix(loss=loss.item(),totalLoss =totalLoss/len(trainDL), TrainAccuracyPerBatch_valence= 100.*totaltrainAccuracybatch_valence, TrainAccuracyPerBatch_arousal = 100.*totaltrainAccuracybatch_arousal)      
            model.eval()
            with torch.no_grad():
                  with tqdm(valDL, unit='batch') as tepoch:
                        total = 0
                        correct_valence = 0
                        correct_arousal = 0
                        val_loss = 0
                        for images,lab in tepoch:
                            images = images.to(device)
                            labelsv = lab.to(device)
                            outval = (model(images.float()))
                            predindexval = (outval>=0.5).float()
                            total += labelsv.size(0)
                            correct_valence += (predindexval[:,0] == labelsv[:,0]).sum().item()
                            correct_arousal += (predindexval[:,1] == labelsv[:,1]).sum().item()
                            lossv = criterion(outval,labelsv.float())
                            val_loss+= lossv.item()
        print('Epoch [{}], Loss_t: {:.4f},Loss_v: {:.4f}, train_acc_v:{:.4f},train_acc_a:{:.4f}, val_acc_v:{:.4f},val_acc_a:{:.4f}'.format(epoch, totalLoss/len(trainDL), val_loss/len(valDL), 100.*totaltrainAccuracybatch_valence, 100.*totaltrainAccuracybatch_arousal,
                                                                                                                           100.*correct_valence/total, 100*correct_arousal/total))
        tL.append(totalLoss/len(trainDL))
        vL.append(val_loss/len(valDL))
        tAV.append(100.*totaltrainAccuracybatch_valence)
        tAA.append(100.*totaltrainAccuracybatch_arousal)
        vAV.append( 100.*correct_valence/total)
        vAC.append(100*correct_arousal/total)
    trainLoss[str(k)] = tL
    valLoss[str(k)] = vL
    trainAcc[str(k)] = [tAV, tAA]
    valAcc[str(k)] = [vAV,vAC]

scipy.io.savemat('StratifiedKF.mat',{'tloss': tL,
                                     'vloss': vL,
                                     'tAV': tAV,
                                     'tAA':tAA,
                                     'vAV':vAV,
                                     'vAA':vAC})
