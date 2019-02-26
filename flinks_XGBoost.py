#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:04:58 2019

@author: Shariful
"""


from __future__ import division
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# Set seed for reproducibility
SEED = 1

data_dir = '/Users/Shariful/Downloads/Flinks'

df_loan = pd.read_csv(data_dir + '/challenge_train.csv')
"""
columns:
    Index([u'LoanFlinksId', u'LoanAmount', u'LoanDate', u'TrxDate',
       u'DaysBeforeRequest', u'Debit', u'Credit', u'Amount', u'Balance',
       u'IsDefault'],
      dtype='object')
    
    
    isdefault: 983923 (0.2386463135233307)
     
"""
features_names = ['LoanAmount', 'Debit', 'Credit', 'Balance', 'IsDefault']

df_loan = df_loan[features_names]

y = df_loan['IsDefault'].values
X = df_loan.drop('IsDefault', axis=1).values

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, \
                                                    random_state=42, stratify=y)

xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)

xg_cl.fit(X_train, y_train)

# Compute predicted probabilities: y_pred_prob
y_pred_prob = xg_cl.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

#preds = xg_cl.predict_proba(X_test)
#
#accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
#print("accuracy: %f" % (accuracy))