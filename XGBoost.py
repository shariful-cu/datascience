#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 23:40:23 2019

@author: Shariful
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data_dir = '/Users/Shariful/Documents/GitHubRepo/Datasets/predictive_foundation'

df_donation = pd.read_csv(data_dir + '/basetable_ex2_4.csv')


y = df_donation['target'].values
X = df_donation.drop('target', axis=1).values





X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123)
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)
xg_cl.fit(X_train, y_train)
preds = xg_cl.predict(X_test)

accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))


"""
Common tree tunable parameters
learning rate: learning rate/eta
gamma: min loss reduction to create new tree split
lambda: L2 reg on leaf weights
alpha: L1 reg on leaf weights
max_depth: max depth per tree
subsample: % samples used per tree
colsample_bytree: % features used per tree
"""

# ==============gridsearchcv for XGBoost=========

#In [1]: import pandas as pd
#In [2]: import xgboost as xgb
#In [3]: import numpy as np
#In [4]: from sklearn.model_selection import GridSearchCV
#In [5]: housing_data = pd.read_csv("ames_housing_trimmed_processed.csv")
#In [6]: X, y = housing_data[housing_data.columns.tolist()[:-1]],
#...: housing_data[housing_data.columns.tolist()[-1]
#In [7]: housing_dmatrix = xgb.DMatrix(data=X,label=y)
#In [8]: gbm_param_grid = {
#...: 'learning_rate': [0.01,0.1,0.5,0.9],
#...: 'n_estimators': [200],
#...: 'subsample': [0.3, 0.5, 0.9]}
#In [9]: gbm = xgb.XGBRegressor()
#In [10]: grid_mse = GridSearchCV(estimator=gbm,
#...: param_grid=gbm_param_grid,
#...: scoring='neg_mean_squared_error'
#, cv=4, verbose=1)
#In [11]: grid_mse.fit(X, y)
#In [12]: print("Best parameters found: "
#,grid_mse.best_params_)
#Best parameters found: {'learning_rate': 0.1,
#'n_estimators': 200,
#'subsample': 0.5}
#In [13]: print("Lowest RMSE found: "
#, np.sqrt(np.abs(grid_mse.best_score_)))
#Lowest RMSE found: 28530.1829341




#random_search = RandomizedSearchCV(xgb, param_distributions=params, \
#                                   n_iter=param_comb, scoring='roc_auc', \
#                                   n_jobs=4, cv=skf.split(X,Y), verbose=3, \
#                                   random_state=1001 )



##=========XGBoost cv with DMatrixx=========
##xgb.DMatrix(data=churn_data.iloc[:,:-1],
##label=churn_data.month_5_still_here)
#
## Create the DMatrix: churn_dmatrix
#churn_dmatrix = xgb.DMatrix(data=X, label=y)
#
## Create the parameter dictionary: params
#params={"objective":"binary:logistic", "max_depth":5}
#
#cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=3, \
#                    num_boost_round=100, metrics="auc", as_pandas=True, seed=123)
#
## Print cv_results
#print(cv_results)
#
## Print the AUC
#print((cv_results["test-auc-mean"]).iloc[-1])