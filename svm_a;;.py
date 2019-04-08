# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:07:56 2019

@author: gungor2
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

from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, KFold
import math
import lightgbm as lgb
from lightgbm import LGBMModel,LGBMClassifier
from statistics import mean 
from keras.models import load_model


from sklearn.preprocessing import StandardScaler

# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense



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

learning_rates = [1]
num_leavess = [2]
sub_samples = [0.3]
col_samples = [0.025]
auc_all = []
auc_max_f = 0

for LR in learning_rates:
    for NL in num_leavess:
        for SS in sub_samples:
            for CS in col_samples:
                

                
                acc_val_f = []
                auc_max_v = 0

                for fold_, (trn_idx, val_idx) in enumerate(folds.split(feature_tr.values, target_tr.values)):
                        print("Fold {}".format(fold_))
                        
                       
                        feature_tr_tr = feature_tr.iloc[trn_idx,:]
                        
                        scaler = StandardScaler().fit(feature_tr_tr)
                        
                        feature_tr_tr = scaler.transform(feature_tr_tr)
                        
                        label_tr=target_tr.iloc[trn_idx]
                        
                        

# Scale the train set

# Scale the test set    class_weight={1: 10}
                        feature_vl = feature_tr.iloc[val_idx,:]
                        feature_vl = scaler.transform(feature_vl)
                    
                        label_vl=target_tr.iloc[val_idx]
                        
                        svmm =sklearn.svm.SVC(kernel = 'linear',class_weight={1: sum(flag0)/sum(flag1)},probability = True,verbose=1)
                        svmm.fit(feature_tr_tr,label_tr)
                        pre = svmm.predict(feature_vl)
                        
                        #a = clf.predict(feature_tr.iloc[trn_idx,:], num_iteration=clf.best_iteration)
                        
                        
                        
                
                        a =  svmm.predict_proba(feature_vl)
                        auc_temp = roc_auc_score(label_vl, a)
                        acc_val_f.append(auc_temp)
                        print(auc_temp)
                        if auc_temp>auc_max_v:
                            auc_max_v = auc_temp
                            clf_f_best = svmm
                            
                        
                        #acc_vl = sum(a==target_tr.iloc[trn_idx]) / len(a)
                        
                        
                temp  = mean(acc_val_f)
                if temp>auc_max_f:
                        auc_max_f = temp
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


