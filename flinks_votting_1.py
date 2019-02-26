
from __future__ import division
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:11:46 2019

@author: Shariful
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 13:50:56 2019

@author: Shariful
"""



import pandas as pd

# Import function to compute accuracy
#from sklearn.metrics import accuracy_score
# Import function to split data
from sklearn.model_selection import train_test_split
# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier as KNN
# Import the VotingClassifier meta-model
from sklearn.ensemble import VotingClassifier
#from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Set seed for reproducibility
SEED = 42

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
features_names = ['LoanAmount', 'Debit', 'Credit', 'Balance', 'IsDefault']

df_loan = df_loan[features_names]

y = df_loan['IsDefault'].values
X = df_loan.drop('IsDefault', axis=1).values

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, \
                                                    random_state=42, stratify=y)

lr = LogisticRegression(random_state=SEED)
#knn = KNN()
dt = DecisionTreeClassifier(random_state=SEED)
#rfc = RandomForestClassifier(random_state=SEED)
# Define a list called classifier that contains
# the tuples (classifier_name, classifier)
classifiers = [('Logistic Regression', lr), \
               ('Classification Tree', dt)]

# Instantiate a VotingClassifier 'vc'
vc = VotingClassifier(estimators=classifiers, n_jobs=4, voting='soft')
# Fit 'vc' to the traing set
vc.fit(X_train, y_train)


## Predict test set labels
#y_pred = vc.predict(X_test)
## Evaluate the test-set accuracy of 'vc'
#print('Voting Classifier: {:.3f}'.format(accuracy_score(y_test, y_pred)))


# Compute predicted probabilities: y_pred_prob
y_pred_prob = vc.predict_proba(X_test)[:,1]

#vc.predict_proba

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))














