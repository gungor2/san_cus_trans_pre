# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:26:45 2019

@author: gungor2
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:32:57 2019

@author: SupErman
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 14:56:04 2019

@author: SupErman
"""


# coding: utf-8

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
from sklearn.neural_network import MLPClassifier
import math
import lightgbm as lgb
from sklearn import svm



file_dir = 'D:\\Dropbox\\job_search\\kaggle'
#file_dir = 'C:\\Users\\SupErman\\Dropbox\\job_search\\kaggle'
file_n = 'train.csv'

data = pd.read_csv(file_dir + '\\' + file_n, error_bad_lines=False)
data.describe()


# In[120]:

print(data.groupby('target').mean())
flag = data['target']>1
print(type(flag))
data_temp = data.loc[flag,]
data_temp
data_clean = data.loc[~flag,]
data_clean.describe()
print(data_clean.groupby('target').mean())
data_clean['target'].value_counts()
print(data_clean.shape)
new_data = data_clean.convert_objects(convert_numeric=True)
print(new_data.dropna().shape)
data_clean = new_data.dropna()



# In[48]:

plt.hist(x='target',data=data_clean)
plt.show()


# In[214]:


max_depths = [4]
n_estimators = [10]

flag1 = data_clean['target'] ==1
print(sum(flag1))

flag0 = data_clean['target'] ==0
print(sum(flag0))

data_all_1 = data_clean.loc[flag1,]


data_all_0 = data_clean.loc[flag0,]
data_all_0 = data_clean.sample(frac=1)




data_all_1_ts = data_all_1.sample(frac=0.1)
data_all_1_tr = data_all_1.drop(data_all_1_ts.index)

len_data_1_tr = len(data_all_1) - len(data_all_1_ts)


len_data_0 = len(data_all_0)

n_ite = math.floor(len_data_0/len_data_1_tr)

data_all_0_ts = data_all_0.iloc[n_ite*len_data_1_tr:len_data_0,:]

test = data_all_0_ts.append(data_all_1_ts,ignore_index = True)
n = len(data_clean.columns)
feature_ts = test.iloc[:,2:n]
target_ts = test.iloc[:,1]


param = {
    'bagging_freq': 1,
    'bagging_fraction': 0.75,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.15,
    'learning_rate': 0.01,
    'max_depth': -1,  
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 2,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 1,
    'max_bin' : 400,
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0  # L2 regularization term on weights
}

acc_ts_i = []
acc_ts_c = []

i=0
data_all_0_us = data_all_0.iloc[i*len_data_1_tr:(i+1)*len_data_1_tr,:]


data_combined = data_all_1_tr.append(data_all_0_us,ignore_index = True)
print(data_combined.shape)
data_combined = data_combined.sample(frac=1)

#data_combined = data_clean

train = data_combined
train_tr = train.iloc[:,2:n]
target_tr = train.iloc[:,1]

num_round = 1000000
    
acc_vl_max = 0
folds = KFold(n_splits=5, shuffle=False, random_state=44000)

n = len(train_tr.columns)
cols = train_tr.columns
feature_best = []
feature_best.append('var_0')
feature_best.append('var_1')

acc_all = []
for j in range(n):

    acc_val_max_rf = 0
    for i in range(n):
        print(str(j)+ ' ' + str(i) +  ' out of ' + str(len(cols)))
        feature_temp = cols[i]
        if (feature_temp in feature_best):
            continue
        temp_best = feature_best[:]
        temp_best.append(feature_temp)
        acc= 0 
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_tr.values, target_tr.values)):
            
            
            
            
            feature_tr = train_tr.iloc[trn_idx,:]
            feature_tr = feature_tr[temp_best]
            
            mean_tr_tr = feature_tr.mean()
            std_tr_tr =  feature_tr.std()
            label_tr = target_tr.iloc[trn_idx]
            
            feature_vl = train_tr.iloc[val_idx,:]
            feature_vl = feature_vl[temp_best]
            
            label_vl = target_tr.iloc[val_idx]
            
            
            
            
            
            
            nor_feature_tr = (feature_tr-mean_tr_tr)/std_tr_tr
            
            nor_feature_vl = (feature_vl-mean_tr_tr)/std_tr_tr
            
            #svmm =sklearn.svm.SVC(kernel = 'linear')
            svmm = MLPClassifier(solver='adam', alpha=1e-5,
                                 hidden_layer_sizes=(5, 5,2), random_state=1)
            
            svmm.fit(nor_feature_tr,label_tr)
            
            pre = svmm.predict(nor_feature_vl)
            
            acc = acc + sum(pre==label_vl) / len(pre)
        if acc>acc_val_max_rf:
            acc_val_max_rf = acc
            best_fr =  feature_temp
            print(best_fr)
            print(acc_vl_max/5)
            
    acc_all.append(acc_vl_max/5)
    feature_best.append(best_fr)
    print(acc_all)
    print(feature_best)


    