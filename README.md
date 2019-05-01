# san_cus_trans_pre
Santander Customer Transaction Prediction data from a very very interesting kaggle competitions..

The challanges in the data:

-- Imbalanced data sets. The class distribution is 10% vs 90%.

-- High dimension: there are 200 features.

-- Interestingly, all the features are uncorrelated which shows that they were preprocessed with some kind of algorithms such as principical component analysis.

-- Shuffling observations within each feature does not affect the accuracy which indicates that the features may be categorical although they look like real numbers, probably due to the principal component analysis.
