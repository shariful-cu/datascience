#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 03:40:16 2019

@author: Shariful
"""












import pandas as pd

# Import function to compute accuracy
from sklearn.metrics import accuracy_score
# Import function to split data
from sklearn.model_selection import train_test_split
# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
# Import the VotingClassifier meta-model
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# Set seed for reproducibility
SEED = 1

data_dir = '/Users/Shariful/Documents/GitHubRepo/Datasets/predictive_foundation'

df_donation = pd.read_csv(data_dir + '/basetable_ex2_4.csv')

y = df_donation['target'].values
X = df_donation.drop('target', axis=1).values

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= SEED)
# Instantiate individual classifiers
lr = LogisticRegression(random_state=SEED)
knn = KNN()
dt = DecisionTreeClassifier(random_state=SEED)
rfc = RandomForestClassifier(random_state=SEED)
# Define a list called classifier that contains
# the tuples (classifier_name, classifier)
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours' , knn), \
               ('Classification Tree', dt), ('Random Forest Classifier', rfc)]

# Iterate over the defined list of tuples containing the classifiers
for clf_name, clf in classifiers:
    #fit clf to the training set
    clf.fit(X_train, y_train)
    # Predict the labels of the test set
    y_pred = clf.predict(X_test)
    # Evaluate the accuracy of clf on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy_score(y_test, y_pred)))


# Instantiate a VotingClassifier 'vc'
vc = VotingClassifier(estimators=classifiers)
# Fit 'vc' to the traing set
vc.fit(X_train, y_train)
# Predict test set labels
y_pred = vc.predict(X_test)
# Evaluate the test-set accuracy of 'vc'
print('Voting Classifier: {:.3f}'.format(accuracy_score(y_test, y_pred)))




## =========GridSearchCV for RandomForrestClassifier===
#rfc=RandomForestClassifier(random_state=42)
#param_grid = { 
#    'n_estimators': [200, 500],
#    'max_features': ['auto', 'sqrt', 'log2'],
#    'max_depth' : [4,5,6,7,8],
#    'criterion' :['gini', 'entropy']
#}
#CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
#CV_rfc.fit(x_train, y_train)
#CV_rfc.best_params_
#
#rfc1=RandomForestClassifier(random_state=42, max_features='auto', \
#                            n_estimators= 200, max_depth=8, criterion='gini')
#rfc1.fit(x_train, y_train)
#
#pred=rfc1.predict(x_test)
#
##writing to csv
#op=pd.DataFrame(test['PassengerId'])
#op['Survived']=op_rf
#
#op.to_csv("op_rf.csv", index=False)




