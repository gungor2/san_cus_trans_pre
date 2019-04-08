# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:36:53 2019

@author: gungor2
"""


# In[46]:

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
from lightgbm import LGBMModel,LGBMClassifier
from statistics import mean 





file_dir = 'D:\\Dropbox\\job_search\\kaggle'
#file_dir = 'C:\\Users\\SupErman\\Dropbox\\job_search\\kaggle'
#file_n = 'train.csv'
#data = pd.read_csv(file_dir + '\\' + file_n, error_bad_lines=False)
#
#feature_real = data.iloc[:,2:]
#targets = data.iloc[:,1]


file_n = 'test.csv'

data = pd.read_csv(file_dir + '\\' + file_n, error_bad_lines=False)
n_r = len(data.columns)
feature_real = data.iloc[:,1:n_r]




clf0 = lgb.Booster(model_file='clf0_a2-8.txt')
a0 = clf0.predict(feature_real) 
print(0)

clf1 = lgb.Booster(model_file='clf1_a2-8.txt')
a1 = clf1.predict(feature_real) 
print(1)
clf2 = lgb.Booster(model_file='clf2_a2-8.txt')
a2 = clf2.predict(feature_real) 
print(2)

clf3 = lgb.Booster(model_file='clf3_a2-8.txt')
a3 = clf3.predict(feature_real) 
print(3)
#

clf4 = lgb.Booster(model_file='clf4_a2-8.txt')
a4 = clf4.predict(feature_real) 
print(0)

clf5 = lgb.Booster(model_file='clf5_a2-8.txt')
a5 = clf5.predict(feature_real) 
print(1)

clf6 = lgb.Booster(model_file='clf6_a2-8.txt')
a6 = clf6.predict(feature_real) 
print(2)

clf7 = lgb.Booster(model_file='clf7_a2-8.txt')
a7 = clf7.predict(feature_real) 
print(3)
#

clf8 = lgb.Booster(model_file='clf8_a2-8.txt')
a8 = clf8.predict(feature_real) 
print(3)
#
#pre = (a0 + a1 +a2 +a3 + a4 + a5 + a6 + a7 + a8)/9


bst = lgb.Booster(model_file='best_grid.txt')

pre = bst.predict(feature_real) 

pre = (a0 + a1 +a2 +a3 + a4 + a5 + a6 + a7 + a8 + pre)/10
#auc_temp = roc_auc_score(targets, pre)


sub_df = pd.DataFrame({"ID_code":data["ID_code"].values})
#sub_df["target"] = a0
sub_df["target"] = pre
sub_df.to_csv("combined_.csv", index=False)