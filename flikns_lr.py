from __future__ import division
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 13:50:56 2019

@author: Shariful
"""

import pandas as pd
import numpy as np

# Import function to compute accuracy

# Import function to split data

## Import models
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_auc_score


# Import the VotingClassifier meta-model

import seaborn as sns
from datetime import datetime





sns.set()

# Set seed for reproducibility
SEED = 1


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))






data_dir = '/Users/Shariful/Downloads/Flinks'

df_loan = pd.read_csv(data_dir + '/challenge_train.csv')
"""
columns:
    Index([u'LoanFlinksId', u'LoanAmount', u'LoanDate', u'TrxDate',
       u'DaysBeforeRequest', u'Debit', u'Credit', u'Amount', u'Balance',
       u'IsDefault'],
      dtype='object')
    
    
    isdefault: 983923 (0.2386463135233307)
    
    lg_reg: AUC: 0.621019847568
     
"""
features_names = ['LoanFlinksId', 'LoanAmount', 'Debit', 'Credit', 'Balance', \
                  'IsDefault']

df_loan = df_loan[features_names]



new_mean_features = ['LoanAmount_mean', 'Debit_mean', 'Credit_mean', 'Balance_mean']
new_median_features = ['LoanAmount_median', 'Debit_median', 'Credit_median', 'Balance_median']
new_std_features = ['LoanAmount_std', 'Debit_std', 'Credit_std', 'Balance_std']

#new_feature_names = ['LoanFlinksId' 'LoanAmount_mean', 'LoanAmount_median', 'LoanAmount_std', \
#                     'Debit_mean', 'Debit_median', 'Debit_std', \
#                     'Credit_mean', 'Credit_median', 'Credit_std', \
#                     'Balance_mean', 'Balance_median', 'Balance_std', 'IsDefault']

new_feature_names = ['LoanFlinksId'] + new_mean_features + \
                     new_median_features + new_std_features + ['IsDefault']
                     


profile_ids = list(df_loan.LoanFlinksId.unique())

df_transform = pd.DataFrame(np.empty([len(profile_ids), len(new_feature_names)]))
df_transform.columns = new_feature_names
df_transform['LoanFlinksId'] = np.array(profile_ids)

#df_transform['LoanAmount_mean'] = [df_transform.loc[i, 'LoanAmount_mean'] = \
#             df_loan[df_loan['LoanFlinksId'] == val]['LoanAmount'].mean() for val in profile_ids]


start_time = timer(None)

for i in range(0, 100):

    df_profile = df_loan[df_loan['LoanFlinksId'] == profile_ids[i]]
    
#    df_transform.loc[i, 'LoanFlinksId'] = profile_ids[i]
    df_transform.loc[i, 'IsDefault'] = df_profile.iloc[0,-1]
    
    df_profile = df_profile.drop(['IsDefault', 'LoanFlinksId'], axis=1)
    
    df_transform.loc[i, new_mean_features] = list(df_profile.mean())
#    df_transform.loc[i, new_median_features] = list(df_profile.median())
#    df_transform.loc[i, new_std_features] = list(df_profile.std())
    
timer(start_time)


## features selection
#for i in range(0, len(profile_ids)):
#    
#    val = profile_ids[i]
#    
#    df_transform.loc[i, 'LoanFlinksId'] = val
#    
#    df_transform.loc[i, 'LoanAmount_mean'] = df_loan[df_loan['LoanFlinksId'] == val]['LoanAmount'].mean()
#    df_transform.loc[i, 'LoanAmount_median'] = df_loan[df_loan['LoanFlinksId'] == val]['LoanAmount'].median()
#    df_transform.loc[i, 'LoanAmount_std'] = df_loan[df_loan['LoanFlinksId'] == val]['LoanAmount'].std()
#    
#    df_transform.loc[i, 'Debit_mean'] = df_loan[df_loan['LoanFlinksId'] == val]['Debit'].mean()
#    df_transform.loc[i, 'Debit_median'] = df_loan[df_loan['LoanFlinksId'] == val]['Debit'].median()
#    df_transform.loc[i, 'Debit_std'] = df_loan[df_loan['LoanFlinksId'] == val]['Debit'].std()
#    
#    df_transform.loc[i, 'Credit_mean'] = df_loan[df_loan['LoanFlinksId'] == val]['Credit'].mean()
#    df_transform.loc[i, 'Credit_median'] = df_loan[df_loan['LoanFlinksId'] == val]['Credit'].median()
#    df_transform.loc[i, 'Credit_std'] = df_loan[df_loan['LoanFlinksId'] == val]['Credit'].std()
#    
#    df_transform.loc[i, 'Balance_mean'] = df_loan[df_loan['LoanFlinksId'] == val]['Balance'].mean()
#    df_transform.loc[i, 'Balance_median'] = df_loan[df_loan['LoanFlinksId'] == val]['Balance'].median()
#    df_transform.loc[i, 'Balance_std'] = df_loan[df_loan['LoanFlinksId'] == val]['Balance'].std() 
#    
#    df_transform.loc[i, 'IsDefault'] = df_loan[df_loan['LoanFlinksId'] == val]['IsDefault'].max() 
#    
    

##---------
#
#y = df_transform['IsDefault'].values
#X = df_transform.drop(['IsDefault', 'LoanFlinksId'], axis=1).values
#
#
###apply on hold-on testing
##df_test = pd.read_csv(data_dir + '/challenge_test.csv')
##X_test = df_test[['LoanAmount', 'Debit', 'Credit', 'Balance']].values
#
## Create training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, \
#                                                    random_state=42, stratify=y)
#
#
#
## Create the classifier: logreg
#logreg = LogisticRegression()
#
## Fit the classifier to the training data
##logreg.fit(X, y)
#logreg.fit(X_train, y_train)
#
#
###apply on hold-on testing
##df_test = pd.read_csv(data_dir + '/challenge_test.csv')
##df_test = df_test[features_names] 
# 
#
#
## Compute predicted probabilities: y_pred_prob
#y_pred_prob = logreg.predict_proba(X_test)[:,1]
#
#
###writing to csv
##op=pd.DataFrame(df_test['LoanFlinksId'])
##op['IsDefault']=y_pred_prob
##
##op.to_csv("/Users/Shariful/Desktop/MdShariful_Islam.csv", index=False)
#
## Compute and print AUC score
#print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
#
#
#
###------EDA: Exploring the Gapminder data-------
##sns.heatmap(df_loan.corr(), square=True, cmap='RdYlGn')










