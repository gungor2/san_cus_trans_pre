# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:17:49 2019

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
import math
import lightgbm as lgb
from lightgbm import LGBMModel,LGBMClassifier
import statsmodels.api as sm
from statsmodels.distributions.mixture_rvs import mixture_rvs


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













param = {
    'bagging_freq': 1,
    'bagging_fraction': 0.6,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.15,
    'learning_rate': 0.01,
    'max_depth': -1,  
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 5,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 1,
    'max_bin' : 400
}

acc_ts_i = []
acc_ts_c = []



data_combined = data_clean
print(data_combined.shape)
data_combined = data_combined.sample(frac=1)


columns=['var_12','var_13','var_108','var_126','var_68']
for var in columns:
    print(var)
    hist, bin_edges = np.histogram(data_combined[var], bins=1000, density=True)
    data_combined['test_'+var] = [ hist[np.searchsorted(bin_edges,ele)-1] for ele in data_combined[var] ]


test_fr = 0.2

test = data_combined.sample(frac=test_fr)
train = data_combined.drop(test.index)

#data_combined = data_clean

n = len(data_clean.columns)
feature_tr = train.iloc[:,2:n]
target_tr = train.iloc[:,1]

feature_ts = test.iloc[:,2:n]
target_ts = test.iloc[:,1]

num_round = 1000000
    
acc_vl_max = 0
folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=44000)




lgb_params = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'is_unbalance': True,
    #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
    'num_leaves': 2,  # we should let it be smaller than 2^(max_depth)
    'max_depth': -1,  # -1 means no limit
    'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 255,  # Number of bucketed bin for feature values
    'subsample': 0.3,  # Subsample ratio of the training instance.
    'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0,  # L2 regularization term on weights
    'nthread': 4,
    'verbose': 0,
    'metric':'auc'
}

for fold_, (trn_idx, val_idx) in enumerate(folds.split(feature_tr.values, target_tr.values)):
    print("Fold {}".format(fold_))
    
   
    
    trn_data = lgb.Dataset(feature_tr.iloc[trn_idx,:], label=target_tr.iloc[trn_idx])


    val_data = lgb.Dataset(feature_tr.iloc[val_idx,:], label=target_tr.iloc[val_idx])
    
    num_round = 1000000
    
    clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)

    a = clf.predict(feature_tr.iloc[val_idx,:], num_iteration=clf.best_iteration)  
    
    #a = clf.predict(feature_tr.iloc[trn_idx,:], num_iteration=clf.best_iteration)
    
    
    a[a>=0.5] = 1
    a[a<0.5] = 0


    acc_vl = sum(a==target_tr.iloc[val_idx]) / len(a)

    #acc_vl = sum(a==target_tr.iloc[trn_idx]) / len(a)
    if acc_vl>acc_vl_max:
        acc_vl_max = acc_vl
        print(acc_vl)
        clf0 = clf

    
a0 = clf0.predict(feature_ts, num_iteration=clf.best_iteration) 
a_0 = a0
a0[a_0>=0.5] = 1
a0[a_0<0.5] = 0


acc_rf = sum(a0==target_ts) / len(a0)
acc_ts_i.append(acc_rf)
acc_ts_c.append(acc_rf)
print(acc_ts_i)
print(acc_ts_c)





# In[160]:

#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#
#features = data_clean.iloc[:,2:n]
#x = StandardScaler().fit_transform(features)
#pca = PCA()
#pca.fit(x)
#print(pca.explained_variance_ratio_.cumsum())


