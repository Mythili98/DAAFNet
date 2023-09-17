#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:01:50 2023

@author: desktop
"""

import scipy
import modelsubjectdep_DA
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

class Dataset(Dataset):
    def __init__(self, data, infoL,infoD):
        #data loading
        self.x = data
        self.yL = infoL
        self.yD = infoD
        self.n_samples = data.shape[0]


    def __getitem__(self,index):
        t1 = self.x[index]
        t2L = self.yL[index]
        t2D = self.yD[index]
        t1 = torch.tensor(t1)
        t1 = t1.unsqueeze(0)
        t2L = torch.tensor(t2L)
        t2D = torch.tensor(t2D)
        t2 = [t2L,t2D]
        return (t1,t2)
    
    def __len__(self):
        return self.n_samples

class TestDataset(Dataset):
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
    model = modelsubjectdep_DA.ModelSubDep()
    model = model.to(device)
    print('Fold :', k)
    data = d
    labels = l
    dataV = data[val_indices,:,:]
    labelsV = labels[val_indices,:]
    dataTr = data[train_indices,:,:]
    labelsTr = labels[train_indices,:]
    dataV, dataT, labelsV, labelsT = train_test_split(dataV, labelsV, test_size=0.5, random_state=42)
    labelsST = np.zeros((dataTr.shape[0],2))
    labelsSV = np.zeros((dataV.shape[0],2))
    labelsST[:,0] = 1
    labelsSV[:,1] = 1
    trainDS = Dataset(dataTr, labelsTr, labelsST)
    trainDL = DataLoader(dataset = trainDS, batch_size = 10, shuffle = True)
    
    valDS = Dataset(dataV, labelsV, labelsSV)
    valDL = DataLoader(dataset = valDS, batch_size = 10, shuffle = True)
    
    testDS = TestDataset(dataT,labelsT)
    testDL = DataLoader(dataset = testDS, batch_size = 20, shuffle = True)
    num_epochs = 200
    batchsize = 20
    criterionLabel = nn.BCELoss()
    criterionDomain = nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
    for epoch in range(num_epochs):
        len_dataloader = min(len(trainDL), len(valDL))
        data_source_iter = iter(trainDL)
        data_target_iter = iter(valDL)
        model.train()
        i = 0
        while i < len_dataloader:
            trainAccuracy_valence = 0
            trainAccuracy_arousal = 0
            totalLoss = 0
            p = float(i + epoch * len_dataloader) / num_epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # training model using source data
            data_source = next(data_source_iter)
            images, lab = data_source
            images = images.to(device)
            labL = lab[0]
            labD = lab[1]
            labL = labL.to(device)
            labD = labD.to(device)
            optimizer_ft.zero_grad()
            outputsL, outputsD = (model(images.float(), alpha))
            predindex_label = (outputsL>=0.5).float()
            _,predindex_domain = torch.max(outputsD,dim=1) 
            loss_label = criterionLabel(outputsL,labL.float())
            loss_domain_train = criterionDomain(outputsD,labD.float())
            loss_train = loss_label+loss_domain_train
            data_target = next(data_target_iter)
            t_img, t_y = data_target
            t_img = t_img.to(device)
            t_y = t_y[1]
            t_y = t_y.to(device)
            _,outputTD = (model(t_img.float(), alpha))
            predindex_domain_target = (outputTD>=0.5).float()
            loss_domain_target = criterionDomain(outputTD,t_y.float())
            loss = loss_domain_target+loss_train
            l2_reg = 0.0
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)  # L2 norm of the outvalparameter
            loss += l2_lambda * l2_reg
            loss.backward()
            totalLoss+=loss.item()
            optimizer_ft.step()
            y_valence = (predindex_label[:,0]==labL[:,0]).sum().item()
            batchAccuracy_valence = y_valence/batchsize
            trainAccuracy_valence += y_valence
            y_arousal = (predindex_label[:,1]==labL[:,1]).sum().item()
            batchAccuracy_arousal = y_arousal/batchsize
            trainAccuracy_arousal += y_arousal
            totaltrainAccuracybatch_valence = trainAccuracy_valence/(len(trainDL)*batchsize)
            totaltrainAccuracybatch_arousal = trainAccuracy_arousal/(len(trainDL)*batchsize)
            i += 1
        model.eval()
        with torch.no_grad():
              with tqdm(testDL, unit='batch') as tepoch:
                    total = 0
                    correct_valence = 0
                    correct_arousal = 0
                    val_loss = 0
                    for images,lab in tepoch:
                        images = images.to(device)
                        labelsv = lab.to(device)
                        outval,_ = (model(images.float(),alpha))
                        predindexval = (outval>=0.5).float()
                        total += labelsv.size(0)
                        correct_valence += (predindexval[:,0] == labelsv[:,0]).sum().item()
                        correct_arousal += (predindexval[:,1] == labelsv[:,1]).sum().item()
                        lossv = criterionLabel(outval,labelsv.float())
                        val_loss+= lossv.item()
        print('Epoch [{}], Loss_t: {:.4f},Loss_v: {:.4f}, train_acc_v:{:.4f},train_acc_a:{:.4f}, val_acc_v:{:.4f},val_acc_a:{:.4f}'.format(epoch, totalLoss/len(trainDL), val_loss/len(valDL), 100.*totaltrainAccuracybatch_valence, 100.*totaltrainAccuracybatch_arousal,
                                                                                                                                           100.*correct_valence/total, 100*correct_arousal/total))
        tL.append(totalLoss/len(trainDL)*2)
        vL.append(val_loss/len(testDL))
        tAV.append(100.*totaltrainAccuracybatch_valence)
        tAA.append(100.*totaltrainAccuracybatch_arousal)
        vAV.append( 100.*correct_valence/total)
        vAC.append(100*correct_arousal/total)
    trainLoss[str(k)] = tL
    valLoss[str(k)] = vL
    trainAcc[str(k)] = [tAV, tAA]
    valAcc[str(k)] = [vAV,vAC]

tn = np.array(list(trainLoss.values()))
vn = np.array(list(valLoss.values()))
ta = np.array(list(trainAcc.values()))
va = np.array(list(valAcc.values()))
scipy.io.savemat('StratifiedKF_DA.mat',{'tloss': tn ,
                                     'vloss': vn,
                                     'tA': ta,
                                     'vA':va})       