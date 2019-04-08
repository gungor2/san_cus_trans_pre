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



file_dir = 'D:\\Dropbox\\job_search\\kaggle'
#file_dir = 'C:\\Users\\SupErman\\Dropbox\\job_search\\kaggle'
file_n = 'test.csv'

data = pd.read_csv(file_dir + '\\' + file_n, error_bad_lines=False)
n_r = len(data.columns)
feature_real = data.iloc[:,1:n_r]

#bst = lgb.Booster(model_file='clf0.txt')

a0 = clf0.predict(feature_real) 
print(0)
a1 = clf1.predict(feature_real) 
print(1)
a2 = clf2.predict(feature_real) 
print(2)
a3 = clf3.predict(feature_real) 
print(3)
#
a4 = clf4.predict(feature_real) 
print(0)
a5 = clf5.predict(feature_real) 
print(1)
a6 = clf6.predict(feature_real) 
print(2)
a7 = clf7.predict(feature_real) 
print(3)
#
a8 = clf8.predict(feature_real) 
print(3)

clf0.save_model('clf0_a2-8.txt')
clf1.save_model('clf1_a2-8.txt')
clf2.save_model('clf2_a2-8.txt')
clf3.save_model('clf3_a2-8.txt')
clf4.save_model('clf4_a2-8.txt')
clf5.save_model('clf5_a2-8.txt')
clf6.save_model('clf6_a2-8.txt')
clf7.save_model('clf7_a2-8.txt')
clf8.save_model('clf8_a2-8.txt')





#a0[a0>=0.50] = 1
#a0[a0<0.5] = 0

#a1[a1>=0.50] = 1
#a1[a1<0.5] = 0
#
#a2[a2>=0.50] = 1
#a2[a2<0.5] = 0
#
#a3[a3>=0.50] = 1
#a3[a3<0.5] = 0
#
#a4[a4>=0.50] = 1
#a4[a4<0.5] = 0
#
#a5[a5>=0.50] = 1
#a5[a5<0.5] = 0
#
#a6[a6>=0.50] = 1
#a6[a6<0.5] = 0
#
#a7[a7>=0.50] = 1
#a7[a7<0.5] = 0
#
#a8[a8>=0.50] = 1
#a8[a8<0.5] = 0
#
#a_t = (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8)/9
#a_t[a_t>0] = 1
#a_t[a_t==0] = 0



#sub_df = pd.DataFrame({"ID_code":data["ID_code"].values})
#sub_df["target"] = a_t
#sub_df.to_csv("submission_0311219_trial.csv", index=False)

sub_df = pd.DataFrame({"ID_code":data["ID_code"].values})
#sub_df["target"] = a0
sub_df["target"] = (a0 + a1 +a2 +a3 + a4 + a5 + a6 + a7 + a8)/9
sub_df.to_csv("submission_single_ind2.csv", index=False)