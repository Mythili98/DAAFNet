#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:42:33 2023

@author: desktop
"""

import DEAP
import glob

def processData():
    p = 0
    data_total = {}
    labels_total = {}
    for filepath in glob.glob('/home/desktop/Desktop/22104412_Docs/EEG-COGMusic/Datasets/deap/PREPROCESSED/data_preprocessed_matlab/*.mat'):
        #filepath = '/home/desktop/Desktop/22104412_Docs/EEG-COGMusic/Datasets/deap/PREPROCESSED/data_preprocessed_matlab/s01.mat' 
        print('Patient ',p,' Processing!')
        data_total[str(p)],labels_total[str(p)] = DEAP.preprocess_subject_dependent(filepath)
        p = p+1
    return data_total, labels_total

data_total, labels_total = processData()

import numpy as np
data = np.array(list(data_total.values()))