# san_cus_trans_pre
Santander Customer Transaction Prediction data from a very very interesting kaggle competitions..

https://www.kaggle.com/c/santander-customer-transaction-prediction/kernels?sortBy=voteCount&group=everyone&pageSize=20&competitionId=10385

# The challanges in the data:

-- Imbalanced data sets. The class distribution is 10% vs 90%.

-- High dimension: there are 200 features.

-- Interestingly, all the features are uncorrelated which shows that they were preprocessed with some kind of algorithms such as principical component analysis.

-- Shuffling observations within each feature does not affect the accuracy which indicates that the features may be categorical although they look like real numbers, probably due to the principal component analysis.


# Libraries used:

Lightgbm, Keras, Sklearn, Scipy, Numpy, Pandas

# Interesing Observations/Lessons learnt

-- Binning actually worked very well in this data

-- When conducting binning, test and training data was combined which created some sort of leakage. But it worked.

-- Testing data had some "unreal" data points that may impact the results. Therefore, they need to be remove. Here is the excellent kernel that removes the fake test data.

https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split


