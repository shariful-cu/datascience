#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 03:53:03 2019

@author: Shariful
"""

import pandas as pd
import numpy as np
# Impsort function to compute accuracy
from sklearn.metrics import accuracy_score
# Import function to split data
from sklearn.model_selection import train_test_split
# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
# Import the VotingClassifier meta-model
from sklearn.ensemble import VotingClassifier
# Set seed for reproducibility
SEED = 1

data_dir = '/Users/Shariful/Documents/GitHubRepo/Datasets/predictive_foundation'

df_donation = pd.read_csv(data_dir + '/basetable_ex2_4.csv')

y = df_donation['target'].values
X = df_donation.drop('target', axis=1).values




# Import models and utility functions
from sklearn.ensemble import BaggingClassifier
# Set seed for reproducibility
SEED = 1
# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, \
                                                    stratify=y, \
                                                    random_state=SEED)

# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=SEED)
# Instantiate a BaggingClassifier 'bc'
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=1)
# Fit 'bc' to the training set
bc.fit(X_train, y_train)
# Predict test set labels
y_pred = bc.predict(X_test)
# Evaluate and print test-set accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))

y_proba = bc.predict_proba(X_test)[:,1]


from sklearn.metrics import roc_auc_score
tpr, fpr, auc = roc_auc_score()

## writing to csv file
#y_proba=y_proba.reshape(-1,1)
#write_np = np.hstack([np.ones([len(y_proba), 1]), y_proba])
#
#write_df = pd.DataFrame(write_np)
#write_df.columns = ['ProfileID', 'Probability']
#
#write_df.to_csv('/Users/Shariful/Desktop/MdShariful_Islam.csv', index=False, header=False)

