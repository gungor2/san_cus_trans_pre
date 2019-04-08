# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:39:53 2019

@author: gungor2
"""

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
from statistics import mean 





file_dir = 'D:\\Dropbox\\job_search\\kaggle'
#file_dir = 'C:\\Users\\SupErman\\Dropbox\\job_search\\kaggle'
file_n = 'train.csv'

data = pd.read_csv( file_n, error_bad_lines=False)
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

columns=data_combined.columns
for i in range(2,len(columns)):
    var = columns[i]
    print(var)
    flag_temp = data_clean['target'] ==1
    data_t = data_combined[var].loc[flag_temp]
    hist, bin_edges = np.histogram(data_t, bins=10, density=True)
    temp_d = []
    for i in range(len(data_combined[var])):
        ele = data_combined[var].iloc[i]    
        temp = np.searchsorted(bin_edges,ele)
        if temp==0 or temp==len(bin_edges):
            temp = 0.00001
            temp_d.append(temp)
        else:
            temp_d.append(hist[temp-1])

    data_combined['test_'+var] = temp_d


test_fr = 0.2

test = data_combined.sample(frac=test_fr)
train = data_combined.drop(test.index)

#data_combined = data_clean

n = len(data_combined.columns)

feature_tr = train.iloc[:,2:n]
target_tr = train.iloc[:,1]

feature_ts = test.iloc[:,2:n]
target_ts = test.iloc[:,1]

num_round = 1000000
    
acc_vl_max = 0
folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=44000)

learning_rates = [0.1,0.01,0.005]
num_leavess = [2,3,4,5]
sub_samples = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
col_samples = [0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8]
bins = [10,50,100,500,1000]
auc_all = []
auc_max_f = 0

for LR in learning_rates:
    for NL in num_leavess:
        for SS in sub_samples:
            for CS in col_samples:
                
                
                
                lgb_params = {
                    'boosting_type': 'gbdt',
                    'learning_rate': LR,
                    'is_unbalance': False,
                    #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
                    'num_leaves': NL,  # we should let it be smaller than 2^(max_depth)
                    'max_depth': -1,  # -1 means no limit
                    'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
                    'max_bin': 255,  # Number of bucketed bin for feature values
                    'subsample': SS,  # Subsample ratio of the training instance.
                    'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
                    'colsample_bytree': CS,  # Subsample ratio of columns when constructing each tree.
                    'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
                    'subsample_for_bin': 200000,  # Number of samples for constructing bin
                    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
                    'reg_alpha': 0,  # L1 regularization term on weights
                    'reg_lambda': 0,  # L2 regularization term on weights
                    'nthread': 4,
                    'verbose': 0,
                    'scale_pos_weight': sum(flag0)/sum(flag1),
                    'boost_from_average':False,
                    'metric':'auc'
                    }

                
                acc_val_f = []
                auc_max_v = 0

                for fold_, (trn_idx, val_idx) in enumerate(folds.split(feature_tr.values, target_tr.values)):
                        print("Fold {}".format(fold_))
                        
                       
                        feature_tr_tr=feature_tr.iloc[trn_idx,:]
                        feature_tr_vl = feature_tr.iloc[val_idx,:]
                        
                        cols_ = feature_tr_tr.columns
                        for i in range(2,len(cols_)):
                            var = cols_[i]
                            print(var)
                            flag_temp = feature_tr_tr['target'] ==1
                            data_t = feature_tr_tr[var].loc[flag_temp]
                            hist, bin_edges = np.histogram(data_t, bins=10, density=True)
                            
                            temp_d = []
                            for i in range(len(feature_tr_tr[var])):
                                ele = feature_tr_tr[var].iloc[i]    
                                temp = np.searchsorted(bin_edges,ele)
                                if temp==0 or temp==len(bin_edges):
                                    temp = 0.00001
                                    temp_d.append(temp)
                                else:
                                    temp_d.append(hist[temp-1])
                        
                            feature_tr_tr['test_'+var] = temp_d


                            temp_d = []
                            for i in range(len(feature_tr_vl[var])):
                                ele = feature_tr_vl[var].iloc[i]    
                                temp = np.searchsorted(bin_edges,ele)
                                if temp==0 or temp==len(bin_edges):
                                    temp = 0.00001
                                    temp_d.append(temp)
                                else:
                                    temp_d.append(hist[temp-1])
                        
                            feature_tr_vl['test_'+var] = temp_d
                            
                            
                        trn_data = lgb.Dataset(feature_tr_tr, label=target_tr.iloc[trn_idx])

                        val_data = lgb.Dataset(feature_tr_vl, label=target_tr.iloc[val_idx])
                        
                        num_round = 1000000
                        
                        clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000000, early_stopping_rounds = 3000)
                    
                        a = clf.predict(feature_tr.iloc[val_idx,:], num_iteration=clf.best_iteration)  
                        
                        #a = clf.predict(feature_tr.iloc[trn_idx,:], num_iteration=clf.best_iteration)
                        
                        
                        
                
                    
                        auc_temp = roc_auc_score(target_tr.iloc[val_idx], a)
                        acc_val_f.append(auc_temp)
                        if auc_temp>auc_max_v:
                            auc_max_v = auc_temp
                            clf_f_best = clf
                        
                        #acc_vl = sum(a==target_tr.iloc[trn_idx]) / len(a)
                        
                        
                temp  = mean(acc_val_f)
                if temp>auc_max_f:
                        auc_max_f = temp
                        clf_f_best.save_model('best_grid.txt')
                        best_LR = LR
                        best_NL = NL
                        best_SS = SS
                        best_CS = CS
                        
                        
                        
                        
                auc_all.append(temp)
                print( 'LR is ' + str(LR) + ' NL is ' + str(NL) + ' SS is ' + str(SS) + ' CS is ' + str(CS) )
                print(auc_all)
                print( 'Best LR is ' + str(best_LR) + ' Best NL is ' + str(best_NL) + ' Best SS is ' + str(best_SS) + ' Best CS is ' + str(best_CS) )
                print('Best AUC is ', str(max(auc_all)))





# In[160]:

#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#
#features = data_clean.iloc[:,2:n]
#x = StandardScaler().fit_transform(features)
#pca = PCA()
#pca.fit(x)
#print(pca.explained_variance_ratio_.cumsum())


