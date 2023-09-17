#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:53:07 2023

@author: Swathi
"""
#import dataPrep
import modelsubjectdep
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import scipy
#data creation

class TrainDataset(Dataset):
    def __init__(self, data, info):
        #data loading
        self.x = data
        self.y = info
        self.n_samples = data.shape[0]

    def __getitem__(self,index):
        t1 = self.x[index]
        t2 = self.y[index]
        t1 = torch.tensor(t1)
        t2 = torch.tensor(t2)
        t1 = t1.unsqueeze(0)
        return (t1,t2)
    
    def __len__(self):
        return self.n_samples

class ValDataset(Dataset):
    def __init__(self, data, info):
        #data loading
        self.x = data
        self.y = info
        self.n_samples = data.shape[0]

    def __getitem__(self,index):
        t1 = self.x[index]
        t2 = self.y[index]
        t1 = torch.tensor(t1)
        t2 = torch.tensor(t2)
        t1 = t1.unsqueeze(0)
        return (t1,t2)
    
    def __len__(self):
        return self.n_samples


trainLoss={}
valLoss = {}
trainAcc = {}
valAcc = {}
# data_total, labels_total = dataPrep.processData()
data = scipy.io.loadmat('/home/desktop/Desktop/22104412_Docs/EEG-COGMusic/data_s01.mat')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_total = data['data']
labels_total = data['labels']
for p in range(data_total.shape[0]):
    trainLossp = []
    valLossp = []
    trainAccp = []
    valAccp = []
    for k in range(10):
        model = modelsubjectdep.ModelSubDep()
        model = model.to(device)
        print('Fold :', k)
        val_indices = k
        train_indices = list(set(list(range(10))) - set([val_indices]))    
        print(train_indices, val_indices)
        data = data_total[p]
        labels = labels_total[p]
        dataV = data[val_indices,:,:,:]
        labelsV = labels[val_indices,:]
        dataTr = data[train_indices,:,:,:]
        labelsTr = labels[train_indices,:,:]
        dataT = dataTr[0,:,:,:]
        labelsT = labelsTr[0,:]
        for i in range(dataTr.shape[0]-1):
             dataT = np.vstack((dataT,dataTr[i+1,:,:,:]))
             labelsT = np.vstack((labelsT, labelsTr[i+1,:]))
        
        trainDS = TrainDataset(dataT, labelsT)
        trainDL = DataLoader(dataset = trainDS, batch_size = 10, shuffle = True)
        
        valDS = ValDataset(dataV, labelsV)
        valDL = DataLoader(dataset = valDS, batch_size = 10, shuffle = True)
        
        num_epochs = 200
        batchsize = 10
        criterion = nn.BCELoss()
        optimizer_ft = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
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
                    loss.backward()
                    totalLoss+=loss.item()
                    optimizer_ft.step()
                    y_valence = (predindex[:,0]==lab[:,0]).sum().item()
                    y_arousal = (predindex[:,1]==lab[:,1]).sum().item()
                    batchAccuracy_valence = y_valence/batchsize
                    trainAccuracy_valence += y_valence
                    batchAccuracy_arousal = y_arousal/batchsize
                    trainAccuracy_arousal += y_arousal
                    totaltrainAccuracybatch_valence = trainAccuracy_valence/(len(trainDL)*batchsize)
                    totaltrainAccuracybatch_arousal = trainAccuracy_arousal/(len(trainDL)*batchsize)
                    tepoch.set_postfix(loss=loss.item(),totalLoss =totalLoss/len(trainDL), TrainAccuracyPerBatch_valence = 100.*totaltrainAccuracybatch_valence, TrainAccuracyPerBatch_arousal = 100.*totaltrainAccuracybatch_arousal)      
                model.eval()
                with torch.no_grad():
                      with tqdm(valDL, unit='batch') as tepoch:
                            total = 0
                            correct_valence = 0
                            correct_arousal = 0
                            for images,lab in tepoch:
                                images = images.to(device)
                                labels = lab.to(device)
                                outval = (model(images.float()))
                                predindexval = (outval>=0.5).float()
                                total += labels.size(0)
                                correct_valence += (predindexval[:,0] == labels[:,0]).sum().item()
                                correct_arousal += (predindexval[:,1] == labels[:,1]).sum().item()
                                lossv = criterion(outval,labels.float())
                                val_loss = lossv.item()
            print('Epoch [{}], Loss_t: {:.4f}, train_acc_valence:{:.4f}, val_acc_valence:{:.4f}'.format(epoch, totalLoss/len(trainDL), 100.*totaltrainAccuracybatch_valence, 100.*correct_valence/total))
            trainLossp.append(totalLoss)
            valLossp.append(val_loss)
            trainAccp.append([totaltrainAccuracybatch_arousal, totaltrainAccuracybatch_valence])
            valAccp.append([correct_arousal/total, correct_valence/total])
    
    trainLoss[str(p)] = trainLossp
    valLoss[str(p)] = valLossp
    trainAcc[str(p)] = trainAccp
    valAcc[str(p)] = valAccp
    p = p+1

# import scipy

# data_t = []
# label_t = []
# data_temp = list(data_total.values())
# label_temp = list(labels_total.values())
# data_pat = []
# label_pat =[]
# for i in range(len(data_temp)):
#     data_pat.append(np.array(list(data_temp[i].values())))
#     label_pat.append(np.array(list(label_temp[i].values())))
# data_pat = np.array(data_pat)
# label_pat = np.array(label_pat)
# scipy.io.savemat('data_DEAPTest3secs.mat', {'data':data_pat, 'labels': label_pat})
        