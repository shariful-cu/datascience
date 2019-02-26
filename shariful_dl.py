#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 03:14:54 2019

@author: Shariful
"""


#import numpy as np
import os
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
##importing utility files
"""
Result Analysis:
    I also implemented one deep learning model using Keras and evaluate using
    training set.
    
    I got AUC: 63%
"""


#---- initialization -----
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

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.3, \
                                                    random_state=42, stratify=y_train)

#==== end of preprocessing training dataset ==========


# FUNCTION: Convert labels from categorical to numbers incrementally from 0 
#           to number of unique labels
def categorical_to_number(labels):# labels is a numpy array(:,1)
    init_label = 0
    for label in np.unique(labels):
        labels[labels==label] = init_label
        init_label += 1
    
    return labels

lab_train =  categorical_to_number(y_train)   
target = to_categorical(lab_train)

# normalizing predictors
predictors = X_train
scaler = MinMaxScaler()
predictors = scaler.fit_transform(predictors)

#size of input_shape
n_cols = predictors.shape[1]

#get the architecture of CNN
def get_new_model(input_shape = (n_cols,)):
    # Set up the model
    model = Sequential()
    
    # Add the first layer
    model.add(Dense(100, activation = 'relu', input_shape = input_shape))
    
    # Add layer 2
    model.add(Dense(100, activation = 'relu'))
    
   # Add layer 3
    model.add(Dense(100, activation = 'relu'))
    
    # Add the output layer
    model.add(Dense(2, activation = 'softmax'))
    return (model)


early_stopping_monitor = EarlyStopping(patience=2)
model = get_new_model()
    # Compile the model
model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', \
              metrics = ['accuracy'])
    # Fit the model
model.fit(predictors, target, validation_split = 0.3, \
          callbacks = [early_stopping_monitor], epochs=10)
  #save the model
dir_path=os.path.dirname(__file__)
filePath = dir_path + '/gait_dl2_model.h5'
model.save(filePath)
   
#load the model
my_model = load_model(filePath)
   
# normalizing test set
#df_test = test_set.iloc[:,0:-1]
#scaler = MinMaxScaler()
df_test = X_test

# labeling test set
lab_test = categorical_to_number(y_test)

#predict scores on test set
y_pred_prob = model.predict_proba(df_test)[:,1]
print("dl, AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))