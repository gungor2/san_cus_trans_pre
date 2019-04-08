# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 21:39:13 2019

@author: gungor2
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 21:26:59 2019

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
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC







file_dir = 'D:\\Dropbox\\job_search\\kaggle'
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

# Define the scaler 


# In[215]:



tr_frac = 0.80
test_frac = 0.005
va_frac = 0.19


df = data_combined

n = len(df.columns)

cols = list(df.columns)
cols = cols[2:(len(cols))]

for j in range(1):
    print('testing equals to ' + str(j))
    
    ## training
    training = df.sample(frac = tr_frac,random_state = j+1)
    
   

    


    
    ## testing data
    testing = df.drop(training.index)
    feature_ts = testing.iloc[:,2:n]
    target_ts = testing.iloc[:,1]
    
    acc_val_max_rf = 0
    
    for z in range(1):
        print('valid equals to ' + str(z))
        
        #validation sets
        validation = training.sample(frac = va_frac, random_state = z+1)
        feature_vl = validation.iloc[:,2:n]
        target_vl = validation.iloc[:,1]
        
        #training weights
        training_model = training.drop(validation.index)
        feature_tr = training_model.iloc[:,2:n]
        target_tr = training_model.iloc[:,1]
        
        scaler = StandardScaler().fit(feature_tr)

# Scale the train set
        feature_tr = scaler.transform(feature_tr)

# Scale the test set
        feature_vl = scaler.transform(feature_vl) 
        
        estimator = SVC(kernel="linear")
        selector = RFECV(estimator, step=1, cv=5)
        selector = selector.fit(feature_tr, target_tr)
        

                        
                        
                        
    
    


# In[160]:

#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#
#features = data_clean.iloc[:,2:n]
#x = StandardScaler().fit_transform(features)
#pca = PCA()
#pca.fit(x)
#print(pca.explained_variance_ratio_.cumsum())


44