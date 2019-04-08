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


file_dir = 'D:\\Dropbox\\job_search\\kaggle'
#file_dir = 'C:\\Users\\SupErman\\Dropbox\\job_search\\kaggle'
file_n = 'train.csv'
data = pd.read_csv(file_dir + '\\' + file_n, error_bad_lines=False)
#
feature_real = data.iloc[:,2:]
targets = data.iloc[:,1]

loaded_model = joblib.load('18000_0_svm.pkl')

loaded_model2 = joblib.load('18000_1_svm.pkl')



scaler = StandardScaler().fit(feature_real)

feature_real = scaler.transform(feature_real)

pre1 = loaded_model.predict_proba(feature_real)

pre1 = pre1[:,1]
pre2 = loaded_model2.predict_proba(feature_real)
pre2 = pre2[:,1]

auc_temp = roc_auc_score(targets, (pre1 + pre2)/2)

print(auc_temp)
file_dir = 'D:\\Dropbox\\job_search\\kaggle'
#file_dir = 'C:\\Users\\SupErman\\Dropbox\\job_search\\kaggle'
file_n = 'test.csv'
loaded_model = joblib.load('18000_0_svm.pkl')

loaded_model2 = joblib.load('18000_1_svm.pkl')
data = pd.read_csv(file_dir + '\\' + file_n, error_bad_lines=False)
n_r = len(data.columns)

feature_real = data.iloc[:,1:n_r]

scaler = StandardScaler().fit(feature_real)
feature_real = scaler.transform(feature_real)


pre1 = loaded_model.predict_proba(feature_real)

pre2 = loaded_model2.predict_proba(feature_real)
pre1 = pre1[:,1]
pre2 = pre2[:,1]
print('here')

sub_df = pd.DataFrame({"ID_code":data["ID_code"].values})
sub_df["target"] = (pre1+pre2)/2
sub_df.to_csv("svm_p1800000_grid.csv", index=False)



pre1_d = loaded_model.decision_function(feature_real)

pre2_d = loaded_model2.decision_function(feature_real)



sub_df = pd.DataFrame({"ID_code":data["ID_code"].values})
sub_df["target"] = (pre1_d+pre2_d)/2
sub_df.to_csv("svm_d18000f0_grid.csv", index=False)