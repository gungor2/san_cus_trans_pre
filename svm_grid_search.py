# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:03:39 2019

@author: gungor2
"""

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
import math
from sklearn import svm

from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, KFold
import math

from statistics import mean 


from sklearn.preprocessing import StandardScaler


from sklearn.externals import joblib



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
kernels = ['rbf','polynomial1','sigmoid','polynomial2','polynomial3','polynomial4']
gammas = [0.01,0.1,1,10,50,100]
degree = [1,2,3,4]
Cs  = gammas
auc_all = []
auc_max_f = 0

for ker in kernels:
    for gamma in gammas:
        for C in Cs:
            
            if ker =='polynomial1':
                ker_i = 'linear'
                degree = 1
            elif ker =='polynomial2':
                ker_i = 'polynomial'
                degree = 2
            elif ker =='polynomial3':
                ker_i = 'polynomial'
                degree = 3
            elif ker =='polynomial4':
                ker_i = 'polynomial'
                degree = 4
            else:
                ker_i = ker
                degree = 1

                
            acc_val_f = []
            auc_max_v = 0

            for fold_, (trn_idx, val_idx) in enumerate(folds.split(feature_tr.values, target_tr.values)):
                    print("Fold {}".format(fold_))
                    
                   
                    feature_tr_tr = feature_tr.iloc[trn_idx,:]
                    
                    
                    scaler = StandardScaler().fit(feature_tr_tr)
                    
                    feature_tr_tr = scaler.transform(feature_tr_tr)
                    
                    label_tr=target_tr.iloc[trn_idx]
                    
                    #feature_tr_tr = feature_tr_tr[1:100,]
                    #label_tr = label_tr[1:100]
                    
                    

# Scale the train set

# Scale the test set    class_weight={1: 10}
                    feature_vl = feature_tr.iloc[val_idx,:]
                    feature_vl = scaler.transform(feature_vl)
                
                    label_vl=target_tr.iloc[val_idx]
                    
                    svmm =sklearn.svm.SVC(kernel = ker_i,class_weight={1: sum(flag0)/sum(flag1)},probability = True,verbose=1,C=C,gamma=gamma,degree=degree)
                    svmm.fit(feature_tr_tr,label_tr)
                    pre = svmm.predict(feature_vl)
                    
                    #a = clf.predict(feature_tr.iloc[trn_idx,:], num_iteration=clf.best_iteration)
                    
                    
                    
            
                    a =  svmm.predict_proba(feature_vl)
                    a1 = a[:,1]
                    auc_temp = roc_auc_score(label_vl, a1)
                    acc_val_f.append(auc_temp)
                    print(auc_temp)
                    if auc_temp>auc_max_v:
                        auc_max_v = auc_temp
                        clf_f_best = svmm
                        
                    
                    #acc_vl = sum(a==target_tr.iloc[trn_idx]) / len(a)
                    
                    
            temp  = mean(acc_val_f)
            if temp>auc_max_f:
                    auc_max_f = temp
                    best_C = C
                    best_ker = ker_i
                    best_gam = gamma
                    joblib.dump(svmm, 'best_grid_svm.pkl') 

                    
                    
                    
                    
            auc_all.append(temp)
            print( 'Kernel is ' + ker_i + ' Gamma is ' + str(gamma) + ' C is ' + str(C) )
            print(auc_all)
            print( 'Best Kernel is ' + best_ker + ' Best Gamma is ' + str(best_gam) + ' Best C is ' + str(best_C)  )
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


