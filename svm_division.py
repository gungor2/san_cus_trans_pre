# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:51:16 2019

@author: gungor2
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:53:12 2019

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

C = 1
gamma = 0.1
ker_i = 'rbf'
degree = 1

acc_ts_i = []
acc_ts_c = []

i=0
data_all_0_us = data_all_0.iloc[i*len_data_1_tr:(i+1)*len_data_1_tr,:]


data_combined = data_all_1_tr.append(data_all_0_us,ignore_index = True)
print(data_combined.shape)
data_combined = data_combined.sample(frac=1)

#data_combined = data_clean

train = data_combined
feature_tr = train.iloc[:,2:n]
target_tr = train.iloc[:,1]

num_round = 1000000
    
auc_max_v = 0
folds = StratifiedKFold(n_splits=6, shuffle=False, random_state=44000)

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
    
    svmm =sklearn.svm.SVC(kernel = ker_i,probability = True,verbose=1,C=C,gamma=gamma,degree=degree)
    svmm.fit(feature_tr_tr,label_tr)
    

    a =  svmm.predict_proba(feature_vl)
    a1 = a[:,1]
    auc_temp = roc_auc_score(label_vl, a1)
    if auc_temp>auc_max_v:
        auc_max_v = auc_temp
        clf0 = svmm


    
print('here')

i=1
data_all_0_us = data_all_0.iloc[i*len_data_1_tr:(i+1)*len_data_1_tr,:]


data_combined = data_all_1_tr.append(data_all_0_us,ignore_index = True)
print(data_combined.shape)
data_combined = data_combined.sample(frac=1)

#data_combined = data_clean

train = data_combined
feature_tr = train.iloc[:,2:n]
target_tr = train.iloc[:,1]

num_round = 1000000
    
acc_vl_max = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(feature_tr.values, target_tr.values)):
    print("Fold {}".format(fold_))
    
    trn_data = lgb.Dataset(feature_tr.iloc[trn_idx,:], label=target_tr.iloc[trn_idx])


    val_data = lgb.Dataset(feature_tr.iloc[val_idx,:], label=target_tr.iloc[val_idx])
    
    num_round = 1000000
    
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    
    a = clf.predict(feature_tr.iloc[val_idx,:], num_iteration=clf.best_iteration)  
    
    #a = clf.predict(feature_tr.iloc[trn_idx,:], num_iteration=clf.best_iteration)
    
    
    a[a>=0.5] = 1
    a[a<0.5] = 0


    acc_vl = sum(a==target_tr.iloc[val_idx]) / len(a)
    
    #acc_vl = sum(a==target_tr.iloc[trn_idx]) / len(a)
    if acc_vl>acc_vl_max:
        acc_vl_max = acc_vl
        clf1 = clf

    
a1 = clf1.predict(feature_ts, num_iteration=clf.best_iteration) 
a1 = a1
a1[a1>=0.5] = 1
a1[a1<0.5] = 0



acc_rf = sum(a1==target_ts) / len(a1)
acc_ts_i.append(acc_rf)

a_t = (a1+a0)/2
a_t[a_t>=0.5] = 1
a_t[a_t<0.5] = 0
acc_rf = sum(a_t==target_ts) / len(a1)
acc_ts_c.append(acc_rf)

print(acc_ts_i)
print(acc_ts_c)

i=2
data_all_0_us = data_all_0.iloc[i*len_data_1_tr:(i+1)*len_data_1_tr,:]


data_combined = data_all_1_tr.append(data_all_0_us,ignore_index = True)
print(data_combined.shape)
data_combined = data_combined.sample(frac=1)

#data_combined = data_clean

train = data_combined
feature_tr = train.iloc[:,2:n]
target_tr = train.iloc[:,1]

num_round = 1000000
    
acc_vl_max = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(feature_tr.values, target_tr.values)):
    print("Fold {}".format(fold_))
    
    trn_data = lgb.Dataset(feature_tr.iloc[trn_idx,:], label=target_tr.iloc[trn_idx])


    val_data = lgb.Dataset(feature_tr.iloc[val_idx,:], label=target_tr.iloc[val_idx])
    
    num_round = 1000000
    
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    
    a = clf.predict(feature_tr.iloc[val_idx,:], num_iteration=clf.best_iteration)  
    
    #a = clf.predict(feature_tr.iloc[trn_idx,:], num_iteration=clf.best_iteration)
    
    
    a[a>=0.5] = 1
    a[a<0.5] = 0


    acc_vl = sum(a==target_tr.iloc[val_idx]) / len(a)
    
    #acc_vl = sum(a==target_tr.iloc[trn_idx]) / len(a)
    
    if acc_vl>acc_vl_max:
        acc_vl_max = acc_vl
        clf2 = clf

    
a2 = clf2.predict(feature_ts, num_iteration=clf.best_iteration) 
a2 = a2
a2[a2>=0.5] = 1
a2[a2<0.5] = 0




acc_rf = sum(a2==target_ts) / len(a2)
acc_ts_i.append(acc_rf)

a_t = (a2+a0+a1)/3
a_t[a_t>=0.5] = 1
a_t[a_t<0.5] = 0
acc_rf = sum(a_t==target_ts) / len(a2)
acc_ts_c.append(acc_rf)


print(acc_ts_i)
print(acc_ts_c)


i=3
data_all_0_us = data_all_0.iloc[i*len_data_1_tr:(i+1)*len_data_1_tr,:]


data_combined = data_all_1_tr.append(data_all_0_us,ignore_index = True)
print(data_combined.shape)
data_combined = data_combined.sample(frac=1)

#data_combined = data_clean

train = data_combined
feature_tr = train.iloc[:,2:n]
target_tr = train.iloc[:,1]

num_round = 1000000
    
acc_vl_max = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(feature_tr.values, target_tr.values)):
    print("Fold {}".format(fold_))
    
    trn_data = lgb.Dataset(feature_tr.iloc[trn_idx,:], label=target_tr.iloc[trn_idx])


    val_data = lgb.Dataset(feature_tr.iloc[val_idx,:], label=target_tr.iloc[val_idx])
    
    num_round = 1000000
    
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    
    a = clf.predict(feature_tr.iloc[val_idx,:], num_iteration=clf.best_iteration)  
    
    #a = clf.predict(feature_tr.iloc[trn_idx,:], num_iteration=clf.best_iteration)
    
    
    a[a>=0.5] = 1
    a[a<0.5] = 0


    acc_vl = sum(a==target_tr.iloc[val_idx]) / len(a)
    
    #acc_vl = sum(a==target_tr.iloc[trn_idx]) / len(a)
    
    if acc_vl>acc_vl_max:
        acc_vl_max = acc_vl
        clf3 = clf

    
a3 = clf3.predict(feature_ts, num_iteration=clf.best_iteration) 
a3 = a3
a3[a3>=0.5] = 1
a3[a3<0.5] = 0




acc_rf = sum(a3==target_ts) / len(a3)
acc_ts_i.append(acc_rf)

a_t = (a3+a2+a0+a1)/4
a_t[a_t>=0.5] = 1
a_t[a_t<0.5] = 0
acc_rf = sum(a_t==target_ts) / len(a3)
acc_ts_c.append(acc_rf)


print(acc_ts_i)
print(acc_ts_c)
        
i=4
data_all_0_us = data_all_0.iloc[i*len_data_1_tr:(i+1)*len_data_1_tr,:]


data_combined = data_all_1_tr.append(data_all_0_us,ignore_index = True)
print(data_combined.shape)
data_combined = data_combined.sample(frac=1)

#data_combined = data_clean

train = data_combined
feature_tr = train.iloc[:,2:n]
target_tr = train.iloc[:,1]

num_round = 1000000
    
acc_vl_max = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(feature_tr.values, target_tr.values)):
    print("Fold {}".format(fold_))
    
    trn_data = lgb.Dataset(feature_tr.iloc[trn_idx,:], label=target_tr.iloc[trn_idx])


    val_data = lgb.Dataset(feature_tr.iloc[val_idx,:], label=target_tr.iloc[val_idx])
    
    num_round = 1000000
    
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    
    a = clf.predict(feature_tr.iloc[val_idx,:], num_iteration=clf.best_iteration)  
    
    #a = clf.predict(feature_tr.iloc[trn_idx,:], num_iteration=clf.best_iteration)
    
    
    a[a>=0.5] = 1
    a[a<0.5] = 0


    acc_vl = sum(a==target_tr.iloc[val_idx]) / len(a)
    
    #acc_vl = sum(a==target_tr.iloc[trn_idx]) / len(a)
    
    if acc_vl>acc_vl_max:
        acc_vl_max = acc_vl
        clf4 = clf

    
a4 = clf4.predict(feature_ts, num_iteration=clf.best_iteration) 
a4 = a4
a4[a4>=0.5] = 1
a4[a4<0.5] = 0




acc_rf = sum(a4==target_ts) / len(a4)
acc_ts_i.append(acc_rf)

a_t = (a4+a3+a2+a0+a1)/(i+1)
a_t[a_t>=0.5] = 1
a_t[a_t<0.5] = 0
acc_rf = sum(a_t==target_ts) / len(a3)
acc_ts_c.append(acc_rf)


print(acc_ts_i)
print(acc_ts_c)


i=5
data_all_0_us = data_all_0.iloc[i*len_data_1_tr:(i+1)*len_data_1_tr,:]


data_combined = data_all_1_tr.append(data_all_0_us,ignore_index = True)
print(data_combined.shape)
data_combined = data_combined.sample(frac=1)

#data_combined = data_clean

train = data_combined
feature_tr = train.iloc[:,2:n]
target_tr = train.iloc[:,1]

num_round = 1000000
    
acc_vl_max = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(feature_tr.values, target_tr.values)):
    print("Fold {}".format(fold_))
    
    trn_data = lgb.Dataset(feature_tr.iloc[trn_idx,:], label=target_tr.iloc[trn_idx])


    val_data = lgb.Dataset(feature_tr.iloc[val_idx,:], label=target_tr.iloc[val_idx])
    
    num_round = 1000000
    
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    
    a = clf.predict(feature_tr.iloc[val_idx,:], num_iteration=clf.best_iteration)  
    
    #a = clf.predict(feature_tr.iloc[trn_idx,:], num_iteration=clf.best_iteration)
    
    
    a[a>=0.5] = 1
    a[a<0.5] = 0


    acc_vl = sum(a==target_tr.iloc[val_idx]) / len(a)
    
    #acc_vl = sum(a==target_tr.iloc[trn_idx]) / len(a)
    
    if acc_vl>acc_vl_max:
        acc_vl_max = acc_vl
        clf5 = clf

    
a5 = clf5.predict(feature_ts, num_iteration=clf.best_iteration) 
a5 = a5
a5[a5>=0.5] = 1
a5[a5<0.5] = 0




acc_rf = sum(a5==target_ts) / len(a4)
acc_ts_i.append(acc_rf)

a_t = (a5 + a4+a3+a2+a0+a1)/(i+1)
a_t[a_t>=0.5] = 1
a_t[a_t<0.5] = 0
acc_rf = sum(a_t==target_ts) / len(a3)
acc_ts_c.append(acc_rf)


print(acc_ts_i)
print(acc_ts_c)

i=6
data_all_0_us = data_all_0.iloc[i*len_data_1_tr:(i+1)*len_data_1_tr,:]


data_combined = data_all_1_tr.append(data_all_0_us,ignore_index = True)
print(data_combined.shape)
data_combined = data_combined.sample(frac=1)

#data_combined = data_clean

train = data_combined
feature_tr = train.iloc[:,2:n]
target_tr = train.iloc[:,1]

num_round = 1000000
    
acc_vl_max = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(feature_tr.values, target_tr.values)):
    print("Fold {}".format(fold_))
    
    trn_data = lgb.Dataset(feature_tr.iloc[trn_idx,:], label=target_tr.iloc[trn_idx])


    val_data = lgb.Dataset(feature_tr.iloc[val_idx,:], label=target_tr.iloc[val_idx])
    
    num_round = 1000000
    
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    
    a = clf.predict(feature_tr.iloc[val_idx,:], num_iteration=clf.best_iteration)  
    
    #a = clf.predict(feature_tr.iloc[trn_idx,:], num_iteration=clf.best_iteration)
    
    
    a[a>=0.5] = 1
    a[a<0.5] = 0


    acc_vl = sum(a==target_tr.iloc[val_idx]) / len(a)
    
    #acc_vl = sum(a==target_tr.iloc[trn_idx]) / len(a)
    
    if acc_vl>acc_vl_max:
        acc_vl_max = acc_vl
        clf6 = clf

    
a6 = clf6.predict(feature_ts, num_iteration=clf.best_iteration) 
a6 = a6
a6[a6>=0.5] = 1
a6[a6<0.5] = 0




acc_rf = sum(a6==target_ts) / len(a4)
acc_ts_i.append(acc_rf)

a_t = (a6 + a5 + a4+a3+a2+a0+a1)/(i+1)
a_t[a_t>=0.5] = 1
a_t[a_t<0.5] = 0
acc_rf = sum(a_t==target_ts) / len(a3)
acc_ts_c.append(acc_rf)


print(acc_ts_i)
print(acc_ts_c)

i=7
data_all_0_us = data_all_0.iloc[i*len_data_1_tr:(i+1)*len_data_1_tr,:]


data_combined = data_all_1_tr.append(data_all_0_us,ignore_index = True)
print(data_combined.shape)
data_combined = data_combined.sample(frac=1)

#data_combined = data_clean

train = data_combined
feature_tr = train.iloc[:,2:n]
target_tr = train.iloc[:,1]

num_round = 1000000
    
acc_vl_max = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(feature_tr.values, target_tr.values)):
    print("Fold {}".format(fold_))
    
    trn_data = lgb.Dataset(feature_tr.iloc[trn_idx,:], label=target_tr.iloc[trn_idx])


    val_data = lgb.Dataset(feature_tr.iloc[val_idx,:], label=target_tr.iloc[val_idx])
    
    num_round = 1000000
    
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    
    a = clf.predict(feature_tr.iloc[val_idx,:], num_iteration=clf.best_iteration)  
    
    #a = clf.predict(feature_tr.iloc[trn_idx,:], num_iteration=clf.best_iteration)
    
    
    a[a>=0.5] = 1
    a[a<0.5] = 0


    acc_vl = sum(a==target_tr.iloc[val_idx]) / len(a)
    
    #acc_vl = sum(a==target_tr.iloc[trn_idx]) / len(a)
    
    if acc_vl>acc_vl_max:
        acc_vl_max = acc_vl
        clf7 = clf

    
a7 = clf7.predict(feature_ts, num_iteration=clf.best_iteration) 
a7 = a7
a7[a7>=0.5] = 1
a7[a7<0.5] = 0




acc_rf = sum(a7==target_ts) / len(a4)
acc_ts_i.append(acc_rf)

a_t = (a7 + a6 + a5 + a4+a3+a2+a0+a1)/(i+1)
a_t[a_t>=0.5] = 1
a_t[a_t<0.5] = 0
acc_rf = sum(a_t==target_ts) / len(a3)
acc_ts_c.append(acc_rf)


print(acc_ts_i)
print(acc_ts_c)

i=8
data_all_0_us = data_all_0.iloc[i*len_data_1_tr:(i+1)*len_data_1_tr,:]


data_combined = data_all_1_tr.append(data_all_0_us,ignore_index = True)
print(data_combined.shape)
data_combined = data_combined.sample(frac=1)

#data_combined = data_clean

train = data_combined
feature_tr = train.iloc[:,2:n]
target_tr = train.iloc[:,1]

num_round = 1000000
    
acc_vl_max = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(feature_tr.values, target_tr.values)):
    print("Fold {}".format(fold_))
    
    trn_data = lgb.Dataset(feature_tr.iloc[trn_idx,:], label=target_tr.iloc[trn_idx])


    val_data = lgb.Dataset(feature_tr.iloc[val_idx,:], label=target_tr.iloc[val_idx])
    
    num_round = 1000000
    
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    
    a = clf.predict(feature_tr.iloc[val_idx,:], num_iteration=clf.best_iteration)  
    
    #a = clf.predict(feature_tr.iloc[trn_idx,:], num_iteration=clf.best_iteration)
    
    
    a[a>=0.5] = 1
    a[a<0.5] = 0


    acc_vl = sum(a==target_tr.iloc[val_idx]) / len(a)
    
    #acc_vl = sum(a==target_tr.iloc[trn_idx]) / len(a)
    
    if acc_vl>acc_vl_max:
        acc_vl_max = acc_vl
        clf8 = clf

    
a8 = clf8.predict(feature_ts, num_iteration=clf.best_iteration) 
a8 = a8
a8[a8>=0.5] = 1
a8[a8<0.5] = 0




acc_rf = sum(a8==target_ts) / len(a4)
acc_ts_i.append(acc_rf)

a_t = (a8 + a7 + a6 + a5 + a4+a3+a2+a0+a1)/(i+1)
a_t[a_t>=0.5] = 1
a_t[a_t<0.5] = 0
acc_rf = sum(a_t==target_ts) / len(a3)
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


