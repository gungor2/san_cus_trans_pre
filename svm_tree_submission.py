# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:30:06 2019

@author: gungor2
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:26:06 2019

@author: gungor2
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:52:02 2019

@author: gungor2
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:33:44 2019

@author: gungor2
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 18:12:02 2019

@author: SupErman
"""

import sklearn
from sklearn import ensemble

import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, KFold
import math
import lightgbm as lgb
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import glob

import csv



file_dir = 'D:\\Dropbox\\job_search\\kaggle'
#file_dir = 'C:\\Users\\SupErman\\Dropbox\\job_search\\kaggle'

file_n = 'test.csv'

data = pd.read_csv( file_n, error_bad_lines=False)

file_dir = 'D:\\Dropbox\\job_search\\kaggle'
#file_dir = 'C:\\Users\\SupErman\\Dropbox\\job_search\\kaggle'

file_n = 'train.csv'

data_tr = pd.read_csv( file_n, error_bad_lines=False)

n_r = len(data.columns)

feature_real_tr = data_tr.iloc[:,2:n_r]
feature_real = data.iloc[:,1:n_r]


scaler = StandardScaler().fit(feature_real_tr)
feature_real = scaler.transform(feature_real)


svm_models = glob.glob('*.csv')
counter_l = []

names_f = []
seed = '15_'
name_write_pre_cs = seed + 'svm_trees_csvs.csv'


counter=0


for i in range(len(svm_models)):
    
    name_f = svm_models[i]
    if '_18000_' in name_f and seed in name_f:
        print(name_f)
        names_f.append(name_f)
        name_f_pkl = name_f.replace('csv','pkl')
        svm_t = joblib.load(name_f_pkl)
        cols = pd.read_csv(name_f,header=-1)
        cols = cols.iloc[0,:]
        cols = cols - 2
        cols = cols.tolist()
        
        pre = svm_t.predict_proba(feature_real[:,cols])
        pre = pre[:,1]
        if counter ==0:
            pre_t = pre
            counter = counter + 1
        else:
            pre_t = (pre_t*counter + pre ) / (counter+1)
            counter = counter + 1
        print(counter)
        
        with open(name_write_pre_cs, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(names_f)   
            
        if counter % 10 ==0:
            name_write_pre = str(counter) + '_' + seed +  'svm_trees.csv'
            name_write_pre_c =  seed + 'svm_trees_coun.csv'
            counter_l.append(counter)
            with open(name_write_pre_c, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow((counter_l))
                
        
            
                sub_df = pd.DataFrame({"ID_code":data["ID_code"].values})
                sub_df["target"] = pre_t
                sub_df.to_csv(name_write_pre, index=False)
            counter_l = []