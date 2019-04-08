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




file_dir = 'D:\\Dropbox\\job_search\\kaggle'
#file_dir = 'C:\\Users\\SupErman\\Dropbox\\job_search\\kaggle'

file_n = 'test.csv'

data = pd.read_csv(file_dir + '\\' + file_n, error_bad_lines=False)
n_r = len(data.columns)

feature_real = data.iloc[:,1:n_r]

scaler = StandardScaler().fit(feature_real)
feature_real = scaler.transform(feature_real)


svm_models = glob.glob('*.pkl')

counter=0
for i in range(len(svm_models)):
    
    name_f = svm_models[i]
    if name_f[:5] == '18000':
        svm_t = joblib.load(name_f)
        pre = svm_t.predict_proba(feature_real)
        if counter ==0:
            pre_t = pre
            counter = counter + 1
        else:
            pre_t = (pre_t*counter + pre ) / (counter+1)
            counter = counter + 1
        print(counter)       
            
        


sub_df = pd.DataFrame({"ID_code":data["ID_code"].values})
sub_df["target"] = pre_t
sub_df.to_csv("svm_d18000all_grid.csv", index=False)