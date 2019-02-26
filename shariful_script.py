from __future__ import division
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 21:25:04 2019

@author: Shariful
"""

#---- import necessary modules -----
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

#---- set valutes -----
data_dir = '/Users/Shariful/Downloads/Flinks'
# Set seed for reproducibility
SEED = 42

#=== preprocessing training dataset ==========
df_train = pd.read_csv(data_dir + '/challenge_train.csv')
features_names = ['TrxDate', 'LoanFlinksId', 'LoanAmount', 'Debit', 'Credit', \
                  'Balance', 'IsDefault']
df_train = df_train[features_names]

df_train.index = pd.to_datetime(df_train['TrxDate'])

agg = {'LoanAmount': 'first', 'Debit': 'sum', 'Credit': 'sum', 'Balance': 'last', 'IsDefault': 'first'}

train_groupby = df_train.groupby(['LoanFlinksId']).aggregate(agg).dropna()

#adding new features: difference between the sum of credits and debites
train_groupby['diff_credit_debit'] = train_groupby['Credit'] - train_groupby['Debit']

# training features with labels
y_train = train_groupby['IsDefault'].values 
X_train = train_groupby.drop(['IsDefault'], axis=1).values

#normalizing
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

#[OPTIONAL]: Evaluate model on training set
#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.3, \
#                                                    random_state=42, stratify=y_train)

#==== end of preprocessing training dataset ==========


#======= preprocessing test set ========
df_test = pd.read_csv(data_dir + '/challenge_test.csv')
features_names = ['TrxDate', 'LoanFlinksId', 'LoanAmount', 'Debit', 'Credit', \
                  'Balance']
df_test = df_test[features_names]
agg = {'LoanAmount': 'first', 'Debit': 'sum', 'Credit': 'sum', 'Balance': 'last'}

test_groupby = df_test.groupby(['LoanFlinksId']).aggregate(agg).dropna()
#adding new features: difference between the sum of credits and debites
test_groupby['diff_credit_debit'] = test_groupby['Credit'] - test_groupby['Debit']

# test features
X_test = test_groupby.get_values()

#normalizing
scaler = MinMaxScaler()
X_test = scaler.fit_transform(X_test)
#======= end of preprocessing test set ========

# Create the classifier: logreg (lr)
lr = LogisticRegression(random_state=SEED)

# Create the classifier: KNN
from sklearn.neighbors import KNeighborsClassifier as KNN
knn = KNN()

# Create the classifier: Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=SEED)

# Create the classifier: RandomForest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=SEED)

classifiers = [('Logistic Regression', lr), ('KNeighborsClassifier', knn), \
               ('Classification Tree', dt), ('RandomForestClassifier', rfc)]


#[OPTIONAL]: check individual classifier performances

## Iterate over the defined list of tuples containing the classifiers
#for clf_name, clf in classifiers:
#    #fit clf to the training set
#    clf.fit(X_train, y_train)
#    # Predict the labels of the test set
#    y_pred_prob = clf.predict_proba(X_test)[:,1]
#    # Compute and print AUC score
#    print("{}, AUC: {}".format(clf_name, roc_auc_score(y_test, y_pred_prob)))



# ========ENSEMBLE: Instantiate a VotingClassifier 'vc'
from sklearn.ensemble import VotingClassifier
vc = VotingClassifier(estimators=classifiers, n_jobs=4, voting='soft')
# Fit 'vc' to the traing set
vc.fit(X_train, y_train)

# Compute predicted probabilities: y_pred_prob
y_pred_prob = vc.predict_proba(X_test)[:,1]
#print("votting clf, AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

#writing to csv
scores_csv = pd.DataFrame(test_groupby.index.tolist())
scores_csv['IsDefault']=y_pred_prob
scores_csv.columns = ['LoanFlinksId', 'IsDefault']
scores_csv.to_csv("/Users/Shariful/Desktop/MdShariful_Islam.csv", index=False)




#xxxxxxxxxxxxxxxxxxxxxxx TLDR; ROUGH coding xxxxxxxxxxxxxxxxxxxxxxxxx


## =======Bagging with lr
#from sklearn.ensemble import BaggingClassifier
## Instantiate a classification-tree 'dt'
#dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=SEED)
## Instantiate a BaggingClassifier 'bc'
#bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=4)
## Fit 'bc' to the training set
#bc.fit(X_train, y_train)
## Predict test set labels
#y_pred_prob = bc.predict_proba(X_test)[:,1]
#print("Bagging clf, AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))


##======= tuning lr hyper parameter ==========
#from sklearn.model_selection import GridSearchCV
#import numpy as np
#
#
#
## Create the hyperparameter grid
#c_space = np.logspace(-5, 8, 15)
#param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}
#
## Create the classifier: logreg
#logreg = LogisticRegression()
#
## Instantiate the GridSearchCV object: logreg_cv
#logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
#
## Fit it to the training data
#logreg_cv.fit(X_train, y_train)
#
## Print the optimal parameters and best score
#print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
#print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))
#
##======= End of lr hyperparameter tuning ==========


##======== train and test lr model =========
## Create the classifier: logreg
#logreg = LogisticRegression()
#
## Fit the classifier to the training data
##logreg.fit(X, y)
#logreg.fit(X_train, y_train)
#
## Compute predicted probabilities: y_pred_prob
#y_pred_prob = logreg.predict_proba(X_test)[:,1]
###apply on hold-on testing
##df_test = pd.read_csv(data_dir + '/challenge_test.csv')
##df_test = df_test[features_names] 
# 
##========= End of train and test lr model =========

