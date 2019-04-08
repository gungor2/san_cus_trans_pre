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

import lightgbm as lgb



file_dir = 'D:\\Dropbox\\job_search\\kaggle'
file_dir = 'C:\\Users\\SupErman\\Dropbox\\job_search\\kaggle'
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
data_all_0_us = data_all_0.sample(n=sum(flag1),random_state=1)
print(data_all_0_us.shape)

data_combined = data_all_1.append(data_all_0_us,ignore_index = True)
print(data_combined.shape)
data_combined = data_combined.sample(frac=1)

#data_combined = data_clean


# In[215]:



tr_frac = 0.70
test_frac = 0.15
va_frac = 0.15

n = len(data_combined.columns)
test = data_combined.sample(frac = test_frac,random_state = 0)
feature_ts = test.iloc[:,2:n]
target_ts = test.iloc[:,1]


train = data_combined.drop(test.index)
feature_tr = train.iloc[:,2:n]
target_tr = train.iloc[:,1]


param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,  
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 1
}



oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()
acc_vl =[]
acc_ts = []
acc_tr = []

folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)

#folds = KFold(n_splits=10, shuffle=False, random_state=44000)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(feature_tr.values, target_tr.values)):
    print("Fold {}".format(fold_))
    
    trn_data = lgb.Dataset(feature_tr.iloc[trn_idx,:], label=target_tr.iloc[trn_idx])


    val_data = lgb.Dataset(feature_tr.iloc[val_idx,:], label=target_tr.iloc[val_idx])
    
    num_round = 1000000
    
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    
    
    a = clf.predict(feature_tr.iloc[trn_idx,:], num_iteration=clf.best_iteration)
    a[a>=0.5] = 1
    a[a<0.5] = 0
    
    acc_rf = sum(a==target_tr.iloc[trn_idx]) / len(a)
    acc_tr.append(acc_rf)
    print(acc_tr)
    
    oof[val_idx] = clf.predict(feature_tr.iloc[val_idx,:], num_iteration=clf.best_iteration)
    a = oof[val_idx]
    a[a>=0.5] = 1
    a[a<0.5] = 0
    
    acc_rf = sum(a==target_tr.iloc[val_idx]) / len(a)
    acc_vl.append(acc_rf)
    
    print(acc_vl)
    predictions += clf.predict(feature_ts, num_iteration=clf.best_iteration) / folds.n_splits
    
    a = clf.predict(feature_ts, num_iteration=clf.best_iteration) 
    a[a>=0.5] = 1
    a[a<0.5] = 0
    
    
    acc_rf = sum(a==target_ts) / len(a)
    acc_ts.append(acc_rf)
    
    print(acc_ts)
    
    
    

print("CV score: {:<8.5f}".format(roc_auc_score(target_tr, oof)))
    
    
# In[160]:

#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#
#features = data_clean.iloc[:,2:n]
#x = StandardScaler().fit_transform(features)
#pca = PCA()
#pca.fit(x)
#print(pca.explained_variance_ratio_.cumsum())


