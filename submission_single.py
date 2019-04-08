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



file_dir = 'D:\\Dropbox\\job_search\\kaggle'
#file_dir = 'C:\\Users\\SupErman\\Dropbox\\job_search\\kaggle'
file_n = 'test.csv'

data = pd.read_csv(file_dir + '\\' + file_n, error_bad_lines=False)
n_r = len(data.columns)
feature_real = data.iloc[:,1:n_r]

bst = lgb.Booster(model_file='best_grid.txt')

a0 = bst.predict(feature_real) 
print(0)





sub_df = pd.DataFrame({"ID_code":data["ID_code"].values})
sub_df["target"] = a0
sub_df.to_csv("best_grid.csv", index=False)