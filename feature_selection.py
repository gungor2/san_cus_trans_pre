
# coding: utf-8

# In[46]:

import sklearn
from sklearn import ensemble

import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix





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

#flag1 = data_clean['target'] ==1
#print(sum(flag1))
#
#flag0 = data_clean['target'] ==0
#print(sum(flag0))
#
#data_all_1 = data_clean.loc[flag1,]
#data_all_0 = data_clean.loc[flag0,]
#data_all_0_us = data_all_0.sample(n=sum(flag1),random_state=1)
#print(data_all_0_us.shape)
#
#data_combined = data_all_1.append(data_all_0_us,ignore_index = True)
#print(data_combined.shape)
#data_combined = data_combined.sample(frac=1)

data_combined = data_clean


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
        


        acc_best=[]
        feature_best_i =[]
        
        #for i in range(len(cols)):
        best_fr = []

        for j in range(len(cols)):
            print(str(j) +  ' out of ' + str(len(cols)))

            acc_val_max_rf = 0
            for i in range(len(cols)):
                feature_temp = cols[i]
            
                flag_temp = best_fr == feature_temp
                flag_temp = np.asarray(flag_temp)
       
                if (feature_temp in best_fr):
                    continue
                temp_best = best_fr[:]
                temp_best.append(feature_temp)
                
                fe_tr = feature_tr[temp_best]
                fe_vl = feature_vl[temp_best]
            
                for r_d  in max_depths:
                    for estimator in n_estimators:
                        clf = sklearn.ensemble.RandomForestClassifier(n_estimators=estimator)
                        clf.fit( fe_tr , target_tr)
                        
                        

                        
                        pre = clf.predict(fe_vl)
                        #acc_rf = sum(pre==target_vl) / len(pre)
                        acc_rf = f1_score(target_vl, pre, average="macro")
                        
                        if acc_rf>acc_val_max_rf:
                            acc_val_max_rf = acc_rf
                            d_best = r_d
                            feature_best = feature_temp
                            estimator_best = estimator
                            print(acc_val_max_rf)

            best_fr.append(feature_best)
            acc_best.append(acc_val_max_rf)
            print(best_fr)
            print(acc_best)
                        
                        
                        
    
    


# In[160]:

#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#
#features = data_clean.iloc[:,2:n]
#x = StandardScaler().fit_transform(features)
#pca = PCA()
#pca.fit(x)
#print(pca.explained_variance_ratio_.cumsum())


