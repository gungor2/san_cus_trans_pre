# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:35:11 2019

@author: gungor2
"""

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
import random

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

import csv

from sklearn.preprocessing import StandardScaler


from sklearn.externals import joblib

import time

seedd= 52000
np.random.seed(seedd)
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
    
C = 1
gamma = 0.1
ker_i = 'rbf'
degree = 1
n = len(data_clean.columns)

nn0= 18000


all_time = []
all_acc= [] 
all_acc_tr= []

all_acc_cum= [] 
all_acc_tr_cum= []
mylist = list(range(2,n))
data_all_0_ts = data_all_0.sample(frac=0.1)
data_all_1_ts = data_all_1.sample(frac=0.1)


for j in range(1):
    nn = nn0 + j*2000
    time_v=[]
    acc_v = []
    acc_v_tr = []

    cum_a_ts = []
    cum_a_tr = []
    for i in range(100):
        sample_size = 60
        sorted_sample = [
                mylist[i] for i in sorted(random.sample(range(len(mylist)), sample_size))
                ]
        
        t = time.time()
        data_all_0_tr = data_all_0.drop(data_all_0_ts.index)
        
        
        data_all_1_tr = data_all_1.drop(data_all_1_ts.index)
        
        
        data_all_0_us = data_all_0_tr.sample(n=nn)
        
        data_all_1_us = data_all_1_tr.sample(n=nn)
        
        data_combined = data_all_1_us.append(data_all_0_us,ignore_index = True)
        
        
        
        
        
        test = data_all_0_ts.append(data_all_1_ts,ignore_index = True)
        
        feature_ts = test.iloc[:,sorted_sample]
        target_ts = test.iloc[:,1]
    
        
    
        
        
        data_combined = data_combined.sample(frac=1)
        
        #data_combined = data_clean
        
        train = data_combined
        feature_tr = train.iloc[:,sorted_sample]
        target_tr = train.iloc[:,1]
        
        num_round = 1000000
            
        
        
        feature_tr_tr = feature_tr
        
        
        scaler = StandardScaler().fit(feature_tr_tr)
        
        feature_tr_tr = scaler.transform(feature_tr_tr)
        
        label_tr=target_tr
        
        #feature_tr_tr = feature_tr_tr[1:100,]
        #label_tr = label_tr[1:100]
        
        
        
        # Scale the train set
        
        # Scale the test set    class_weight={1: 10}
        feature_vl = feature_ts
        feature_vl = scaler.transform(feature_vl)
        
        label_vl=target_ts
        
        svmm =sklearn.svm.SVC(kernel = ker_i,probability = True,verbose=0)
        svmm.fit(feature_tr_tr,label_tr)
        name_svm = str(sample_size) +'_' + str(nn) + '_' + str(i) + '_' + str(seedd) + '_svm.pkl'
        name_svm_c = str(sample_size) +'_' + str(nn) + '_' + str(i) + '_' + str(seedd) + '_svm.csv'
        joblib.dump(svmm, name_svm)
        with open(name_svm_c, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow((sorted_sample))
            print(sorted_sample)
        
        
        
        a =  svmm.predict_proba(feature_vl)
        a1 = a[:,1]
        auc_temp = roc_auc_score(label_vl, a1)
        
        if i==0:
            cum_a_ts = a1
        else:
            cum_a_ts = (cum_a_ts * i + a1)/(i+1)
            
        auc_temp_ts_cum = roc_auc_score(label_vl, cum_a_ts)
        
        a =  svmm.predict_proba(feature_tr_tr)
        a1 = a[:,1]
        auc_temp_tr = roc_auc_score(label_tr, a1)
        
        if i==0:
            cum_a_tr = a1
        else:
            cum_a_tr = (cum_a_tr * i + a1)/(i+1)
            
        auc_temp_tr_cum = roc_auc_score(label_tr, cum_a_tr)   
        
        
        elapsed = time.time() - t
        time_v.append(elapsed)
        acc_v.append(auc_temp)
        acc_v_tr.append(auc_temp_tr)
        #print(nn)
        #print(time_v)
        #print(acc_v)
        #print(acc_v_tr)
        #print(auc_temp_tr_cum )
        print(i)
        print(auc_temp_ts_cum)
    all_time.append(mean(time_v))
    all_acc.append(mean(acc_v))
    all_acc_tr.append(mean(acc_v_tr))
    all_acc_cum.append(auc_temp_ts_cum)
    all_acc_tr_cum.append(auc_temp_tr_cum)
    
    print(nn)

