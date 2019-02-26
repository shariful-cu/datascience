#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 21:11:20 2019

@author: Shariful
"""
#import necessary modules
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt 
import seaborn as sns

#global values
data_dir = '/Users/Shariful/Documents/GitHubRepo/Datasets/supervised_ml'
sns.set()

def display_plot(cv_scores, cv_scores_std, alpha_space):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)
    #Standard Error of the Mean: std_error = std/sqrt(N)
    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
    


#====================== Analysis of US house-votes-84 dataset =================

#----load US house-votes-84 dataset ------
df_house_votes = pd.read_csv(data_dir + '/house-votes-84.csv')
#'party' is the target variable
feature_names = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
       'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
       'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']
df_house_votes.columns = feature_names

df_house_votes = df_house_votes.replace('y', 1)
df_house_votes = df_house_votes.replace('n', 0)
df_house_votes = df_house_votes.replace('?', np.nan)
#df_house_votes = df_house_votes.dropna()


##--- Dropping missing data ---
## Convert '?' to NaN
#df_house_votes[df_house_votes == '?'] = np.nan
## Print the number of NaNs
#print(df_house_votes.isnull().sum())
## Print shape of original DataFrame
#print("Shape of Original DataFrame: {}".format(df_house_votes.shape))
## Drop missing values and print shape of new DataFrame
#df_house_votes = df_house_votes.dropna()
## Print shape of new DataFrame
#print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df_house_votes.shape))

#---- SVC with Imputing missing data in a ML Pipeline I and II----
# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
# Instantiate the SVC classifier: clf
clf = SVC()
# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]

#Imputing missing data in a ML Pipeline II
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Create arrays for the features and the target variable
y = df_house_votes['party'].values
X = df_house_votes.drop('party', axis=1).values

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, \
                                                    random_state=42)
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))



##----- Visual EDA -----
#"""
#votes for education:
#    Democrats voted resoundingly against this bill
#votes for satellite:
#    Democrats voted resoundingly in favor of this bill
#votes for missile:
#    Democrats voted resoundingly in favor of this bill
#"""
#plt.figure()
#sns.countplot(x='education', hue='party', data=df_house_votes, palette='RdBu')
#plt.xticks([0,1], ['No', 'Yes'])
#plt.show()
#
#plt.figure()
#sns.countplot(x='satellite', hue='party', data=df_house_votes, palette='RdBu')
#plt.xticks([0,1], ['No', 'Yes'])
#plt.show()
#
#plt.figure()
#sns.countplot(x='missile', hue='party', data=df_house_votes, palette='RdBu')
#plt.xticks([0,1], ['No', 'Yes'])
#plt.show()

##---- k-Nearest Neighbors: Fit ------
#from sklearn.neighbors import KNeighborsClassifier
#
## Create arrays for the features and the response variable
#y = df_house_votes['party'].values
#X = df_house_votes.drop('party', axis=1).values
## Create a k-NN classifier with 6 neighbors
#knn = KNeighborsClassifier(n_neighbors=6)
## Fit the classifier to the data
#knn.fit(X, y)
## k-Nearest Neighbors: Predict
## Predict the labels for the training data X
#y_pred = knn.predict(X)
## Predict and print the label for the new data point X_new
## we don't have new data point yet so skipp it
##new_prediction = ____
#print("Prediction: {}".format(y_pred))





#====================== End of Analysis of US house-votes-84 dataset ==========

#====================== Analysis of MNIST digits dataset ======================

## Import necessary modules
#from sklearn import datasets
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import train_test_split
#
## ------Load the digits dataset: digits--
#digits = datasets.load_digits()

##---EDA------
## Print the keys and DESCR of the dataset
#print(digits.keys)
#print(digits.DESCR)
## Print the shape of the images and data keys
#print(digits.images.shape)
#print(digits.data.shape)
## Display digit 1010
#plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
#plt.show()
##output
#"""
#It looks like the image in question corresponds to the digit '5'. Now, can you 
#build a classifier that can make this prediction not only for this image, but 
#for all the other ones in the dataset? You'll do so in the next exercise!
#"""

##------ Train/Test Split + Fit/Predict/Accuracy ---
#"""
#Result Analysis:
#    Incredibly, this out of the box k-NN classifier with 7 neighbors has 
#    learned from the training data and predicted the labels of the images in 
#    the test set with 98% accuracy, and it did so in less than a second! This 
#    is one illustration of how incredibly useful machine learning techniques 
#    can be.
#"""
## Create feature and target arrays
#X = digits.data
#y = digits.target
## Split into training and test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, \
#                                                    random_state = 42, \
#                                                    stratify = y)

## ----- Create a k-NN classifier with 7 neighbors: knn-----
#knn = KNeighborsClassifier(n_neighbors=7)
## Fit the classifier to the training data
#knn.fit(X_train, y_train)
## Print the accuracy
#print(knn.score(X_test, y_test))

##----- Overfitting and underfitting  -------
#"""
#Result Analysis:
#    It looks like the test accuracy is highest when using 3 and 5 neighbors. 
#    Using 8 neighbors or more seems to result in a simple model that underfits 
#    the data. Now that you've grasped the fundamentals of classification, you 
#    will learn about regression in the next chapter!
#"""
## Setup arrays to store train and test accuracies
#neighbors = np.arange(1, 9)
#train_accuracy = np.empty(len(neighbors))
#test_accuracy = np.empty(len(neighbors))
#
## Loop over different values of k
#for i, k in enumerate(neighbors):
#    # Setup a k-NN Classifier with k neighbors: knn
#    knn = KNeighborsClassifier(n_neighbors=k)
#
#    # Fit the classifier to the training data
#    knn.fit(X_train, y_train)
#    
#    #Compute accuracy on the training set
#    train_accuracy[i] = knn.score(X_train, y_train)
#
#    #Compute accuracy on the testing set
#    test_accuracy[i] = knn.score(X_test, y_test)
#
## Generate plot
#plt.title('k-NN: Varying Number of Neighbors')
#plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
#plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
#plt.legend()
#plt.xlabel('Number of Neighbors')
#plt.ylabel('Accuracy')
#plt.show()

#====================== End of Analysis of MNIST digits dataset ===============



##======================================= Analysis of Iris dataset ============
#
##---- loading data -----
#df_iris = load_iris()
#data_iris = df_iris.data
#iris_lab = df_iris.target
#iris_dataframe = pd.DataFrame(data_iris, columns = df_iris.feature_names)
#
##----EDA---
#plt.style.use('ggplot')
#_ = pd.scatter_matrix(iris_dataframe, c = iris_lab, figsize=[8,8], \
#                      s=150, marker='D')
#
#
##=================================End of Analysis of Iris dataset ============


#========================= Analysis of boston housing data ====================

#"""
#When regression is the best ml model:
#    The target variable here - the number of bike rentals at any given hour - 
#    is quantitative, so this is best framed as a regression problem.
#"""
#from sklearn import linear_model
##----load boston housing data ------
#df_boston = pd.read_csv(data_dir + '/boston.csv')
#
##----Creating feature and target arrays-----
#X = df_boston.drop('MEDV', axis=1).values 
#y = df_boston['MEDV'].values
# 
##----Predicting house value from a single feature--- 
#X_rooms = X[:,5] 
##convet ndarray to single dimension
#y = y.reshape(-1, 1)
#X_rooms = X_rooms.reshape(-1, 1)
#
##----EDA-----
#plt.scatter(X_rooms, y)
#plt.ylabel('Value of house /1000 ($)') 
#plt.xlabel('Number of rooms')
#plt.show();
#
##-----Fi!ing a regression model------
#reg = linear_model.LinearRegression() 
#reg.fit(X_rooms, y)
#prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1, 1)
#plt.scatter(X_rooms, y, color='blue') 
#plt.plot(prediction_space, reg.predict(prediction_space), color='black', \
#          linewidth=3)
#plt.show()

#=========================End of Analysis of boston housing data ==============


#====================== Analysis of Gapminder dataset ========================= 

##----load gapminder data ------
#df_gapminder = pd.read_csv(data_dir + '/gm_2008_region.csv')

##------Create arrays for features and target variable------
#y = df_gapminder['life'].values
#X = df_gapminder['fertility'].values
#
## Print the dimensions of X and y before reshaping
#print("Dimensions of y before reshaping: {}".format(y.shape))
#print("Dimensions of X before reshaping: {}".format(X.shape))
#
## Reshape X and y
#"""
#Notice the differences in shape before and after applying the .reshape() 
#method. Getting the feature and target variable arrays into the right format 
#for scikit-learn is an important precursor to model building.
#"""
#y = y.reshape(-1,1)
#X = X.reshape(-1,1)
#
## Print the dimensions of X and y after reshaping
#print("Dimensions of y after reshaping: {}".format(y.shape))
#print("Dimensions of X after reshaping: {}".format(X.shape))

##------EDA: Exploring the Gapminder data-------
#sns.heatmap(df_gapminder.corr(), square=True, cmap='RdYlGn')

##-------Fit & predict for regression using one feature----
## Import LinearRegression
#from sklearn.linear_model import LinearRegression 
##features and target
#y = df_gapminder['life'].values
#X_fertility = df_gapminder['fertility'].values
#
#y = y.reshape(-1,1)
#X_fertility = X_fertility.reshape(-1,1)
#
##plot X_fertility vs life expectancy (y)
#plt.plot(X_fertility, y, marker='.', linestyle='none')
#plt.xlabel('average fertility')
#plt.ylabel('life expectancy')
#
## Create the regressor: reg
#reg = LinearRegression()
## Create the prediction space
#prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)
#
## Fit the model to the data
#reg.fit(X_fertility, y)
#
## Compute predictions over the prediction space: y_pred
#y_pred = reg.predict(prediction_space)
#
## Print R^2 
#print(reg.score(X_fertility, y))
#
## Plot regression line
#plt.plot(prediction_space, y_pred, color='black', linewidth=3)
#plt.show()

##-------Train/test split for regression using all features--------
## Import necessary modules
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import train_test_split
#
#"""
#Result Analysis:
#    Excellent! Using all features has improved the model score. This makes 
#    sense, as the model has more information to learn from. However, there is 
#    one potential pitfall to this process. Can you spot it? You'll learn about 
#    this as well how to better validate your models in the next video!
#"""
#
##all features and target
#y = df_gapminder['life'].values
#X = df_gapminder.drop(['life', 'Region'], axis=1).values
#
## Create training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
#
## Create the regressor: reg_all
#reg_all = LinearRegression()
#
## Fit the regressor to the training data
#reg_all.fit(X_train, y_train)
#
## Predict on the test data: y_pred
#y_pred = reg_all.predict(X_test)
#
## Compute and print R^2 and RMSE
#print("R^2: {}".format(reg_all.score(X_test, y_test)))
#rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#print("Root Mean Squared Error: {}".format(rmse))

##------5-fold cross-validation----
## Import the necessary modules
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import cross_val_score
#
##all features and target
#y = df_gapminder['life'].values
#X = df_gapminder.drop(['life', 'Region'], axis=1).values
#
## Create a linear regression object: reg
#reg = LinearRegression()
## Compute 5-fold cross-validation scores: cv_scores
#cv_scores = cross_val_score(reg, X, y, cv=5)
#
## Print the 5-fold cross-validation scores
#print(cv_scores)
#
#print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

#---Regularization I: Lasso---
"""
Result Analysis:
    According to the lasso algorithm, it seems like 'child_mortality' is the 
    most important feature when predicting life expectancy.
"""

##all features and target
#y = df_gapminder['life'].values
#X = df_gapminder.drop(['life', 'Region'], axis=1).values
##features names
#df_columns = df_gapminder.drop(['life', 'Region'], axis=1).columns
## Import Lasso
#from sklearn.linear_model import Lasso
#
## Instantiate a lasso regressor: lasso
#lasso = Lasso(alpha=0.4, normalize=True)
## Fit the regressor to the data
#lasso.fit(X, y)
## Compute and print the coefficients
#lasso_coef = lasso.coef_
#print(lasso_coef)
## Plot the coefficients
#plt.plot(range(len(df_columns)), lasso_coef)
#plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
#plt.margins(0.02)
#plt.show()

##---------- Regularization II: Ridge --------
#"""
#Result Analysis:
#    Notice how the cross-validation scores change with different alphas. Which 
#    alpha should you pick? How can you fine-tune your model? You'll learn all 
#    about this in the next chapter!
#"""
## Import necessary modules
#from sklearn.linear_model import Ridge
#from sklearn.model_selection import cross_val_score
##all features and target
#y = df_gapminder['life'].values
#X = df_gapminder.drop(['life', 'Region'], axis=1).values
## Setup the array of alphas and lists to store scores
#alpha_space = np.logspace(-4, 0, 50)
#ridge_scores = []
#ridge_scores_std = []
#
## Create a ridge regressor: ridge
#ridge = Ridge(normalize=True)
#
## Compute scores over range of alphas
#for alpha in alpha_space:
#
#    # Specify the alpha value to use: ridge.alpha
#    ridge.alpha = alpha
#    
#    # Perform 10-fold CV: ridge_cv_scores
#    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
#    
#    # Append the mean of ridge_cv_scores to ridge_scores
#    ridge_scores.append(np.mean(ridge_cv_scores))
#    
#    # Append the std of ridge_cv_scores to ridge_scores_std
#    ridge_scores_std.append(np.std(ridge_cv_scores))
#
## Display the plot
#display_plot(ridge_scores, ridge_scores_std, alpha_space)


##------Exploring categorical features----------
## Create a boxplot of life expectancy per region
#df_gapminder.boxplot('life', 'Region', rot=60)
## Show the plot
#plt.show()

##---convert categorical features using pd.get_dummies() -----
## Create dummy variables: df_region
#df_region = pd.get_dummies(df_gapminder)
## Print the columns of df_region
#print(df_region.columns)
## Create dummy variables with drop_first=True: df_region
#df_region = pd.get_dummies(df_gapminder, drop_first=True)
## Print the new columns of df_region
#print(df_region.columns)

##---- Regression with categorical features ----
## Import necessary modules
#from sklearn.linear_model import Ridge
#from sklearn.model_selection import cross_val_score
#
##all features and target
#y = df_region['life'].values
#X = df_region.drop(['life'], axis=1).values
#
## Instantiate a ridge regressor: ridge
#ridge = Ridge(alpha=0.5, normalize=True)
#
## Perform 5-fold cross-validation: ridge_cv
#ridge_cv = cross_val_score(ridge, X, y, cv=5)
#
## Print the cross-validated scores
#print(ridge_cv)


#====================== End of Analysis of Gapminder dataset ==================


#====================== Analysis of diabetes dataset ==========================
"""
Problem:
    The goal is to predict whether or not a given female patient will contract 
    diabetes based on features such as BMI, age, and number of pregnancies. A 
    target value of 0 indicates that the patient does not have diabetes, while 
    a value of 1 indicates that the patient does have diabetes.

Result Analysis:
    By analyzing the confusion matrix and classification report, you can get a 
    much better understanding of your classifier's performance.
"""
##---import modules------
#from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix

##----load diabetes data ------
#df_diabetes = pd.read_csv(data_dir + '/diabetes.csv')
##all features and target
#y = df_diabetes['diabetes'].values
#X = df_diabetes.drop(['diabetes'], axis=1).values

##------KNN---------
## Create training and test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
## Instantiate a k-NN classifier: knn
#knn = KNeighborsClassifier(n_neighbors=6)
## Fit the classifier to the training data
#knn.fit(X_train, y_train)
## Predict the labels of the test data: y_pred
#y_pred = knn.predict(X_test)
## Generate the confusion matrix and classification report
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))

##------Building a logistic regression model---------
#"""
#Result Analysis:
#    You now know how to use logistic regression for binary classification - 
#    great work! Logistic regression is used in a variety of machine learning 
#    applications and will become a vital part of your data science toolbox.
#"""
#from sklearn.linear_model import LogisticRegression
## Create training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)
#
## Create the classifier: logreg
#logreg = LogisticRegression()
#
## Fit the classifier to the training data
#logreg.fit(X_train, y_train)
#
## Predict the labels of the test set: y_pred
#y_pred = logreg.predict(X_test)
#
## Compute and print the confusion matrix and classification report
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
#
##Plotting an ROC curve
## Import necessary modules
#from sklearn.metrics import roc_curve
#
## Compute predicted probabilities: y_pred_prob
#y_pred_prob = logreg.predict_proba(X_test)[:,1]
#
## Generate ROC curve values: fpr, tpr, thresholds
#fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
#
## Plot ROC curve
#plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr, tpr)
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve')
#plt.show()
#
##AUC computation
## Import necessary modules
#from sklearn.metrics import roc_auc_score
#from sklearn.model_selection import cross_val_score
#
## Compute predicted probabilities: y_pred_prob
#y_pred_prob = logreg.predict_proba(X_test)[:,1]
#
## Compute and print AUC score
#print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
#
## Compute cross-validated AUC scores: cv_auc
#cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')
#
## Print list of AUC scores
#print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))

##--------Hyperparameter tuning with GridSearchCV for LogisticRegression------
## Import necessary modules
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import GridSearchCV
#
## Setup the hyperparameter grid
#c_space = np.logspace(-5, 8, 15)
#param_grid = {'C': c_space}
#
## Instantiate a logistic regression classifier: logreg
#logreg = LogisticRegression()
#
## Instantiate the GridSearchCV object: logreg_cv
#logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
#
## Fit it to the data
#logreg_cv.fit(X, y)
#
## Print the tuned parameters and score
#print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
#print("Best score is {}".format(logreg_cv.best_score_))

##--- Decision Tree with Hyperparameter tuning using RandomizedSearchCV ------
#"""
#Result Analysis:
#    You'll see a lot more of decision trees and RandomizedSearchCV as you 
#    continue your machine learning journey. Note that RandomizedSearchCV will 
#    never outperform GridSearchCV. Instead, it is valuable because it saves on 
#    computation time.
#"""
## Import necessary modules
#from scipy.stats import randint
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.model_selection import RandomizedSearchCV
#
## Setup the parameters and distributions to sample from: param_dist
#param_dist = {"max_depth": [3, None],
#              "max_features": randint(1, 9),
#              "min_samples_leaf": randint(1, 9),
#              "criterion": ["gini", "entropy"]}
#
## Instantiate a Decision Tree classifier: tree
#tree = DecisionTreeClassifier()
#
## Instantiate the RandomizedSearchCV object: tree_cv
#tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
#
## Fit it to the data
#tree_cv.fit(X, y)
#
## Print the tuned parameters and score
#print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
#print("Best score is {}".format(tree_cv.best_score_))

##---- LogisticRegression with the Hold-out set in practice I: Classification --
## Import necessary modules
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import GridSearchCV
#
## Create the hyperparameter grid
#c_space = np.logspace(-5, 8, 15)
#param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}
#
## Instantiate the logistic regression classifier: logreg
#logreg = LogisticRegression()
#
## Create train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
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

##---- ElasticNet with the Hold-out set in practice II: Regression --
## Import necessary modules
#from sklearn.linear_model import ElasticNet
#from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import GridSearchCV
#
## Create train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
#
## Create the hyperparameter grid
#l1_space = np.linspace(0, 1, 30)
#param_grid = {'l1_ratio': l1_space}
#
## Instantiate the ElasticNet regressor: elastic_net
#elastic_net = ElasticNet()
#
## Setup the GridSearchCV object: gm_cv
#gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)
#
## Fit it to the training data
#gm_cv.fit(X_train, y_train)
#
## Predict on the test set and compute metrics
#y_pred = gm_cv.predict(X_test)
#r2 = gm_cv.score(X_test, y_test)
#mse = mean_squared_error(y_test, y_pred)
#print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
#print("Tuned ElasticNet R squared: {}".format(r2))
#print("Tuned ElasticNet MSE: {}".format(mse))

#====================== End of Analysis of diabetes dataset ===================
























