#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 00:25:52 2019

@author: Shariful
"""

#import necessary modules
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
# Import the classification algorithm
from sklearn.tree import DecisionTreeClassifier

sns.set()

data_dir = '/Users/Shariful/Documents/GitHubRepo/Datasets/predictive_foundation'

df_donation = pd.read_csv(data_dir + '/basetable_ex2_4.csv')


# Import the classification algorithm
from sklearn.tree import DecisionTreeClassifier

# Initialize it and call model by specifying the random_state parameter
model = DecisionTreeClassifier(random_state=42)

# Apply a decision tree model to fit features to the target
model.fit(features_train, target_train)