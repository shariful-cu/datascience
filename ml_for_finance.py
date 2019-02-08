#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 22:38:13 2019

ERROR: talib is not working with anaconda. SO try with others

@author: Shariful
"""


#---- import necessary modules -----
import pandas as pd
import talib
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score
#import datetime
#import stat_part1 as utlty_f
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import keras.losses
import tensorflow as tf

#---- global instructions -----
data_dir = '/Users/Shariful/Documents/GitHubRepo/Datasets/ml_for_finance'
sns.set()


# Create loss function
def sign_penalty(y_true, y_pred):
    penalty = 100.
    loss = tf.where(tf.less(y_true * y_pred, 0), \
                     penalty * tf.square(y_true - y_pred), \
                     tf.square(y_true - y_pred))
    return tf.reduce_mean(loss, axis=-1)



# =================analyzing AMD and Spy datasets ======================================
"""

"""

#--loading data and set index as datetime---
lng_df = pd.read_csv(data_dir + '/AAPL.csv')
lng_df.index = pd.to_datetime(lng_df['Date'])

spy_df = pd.read_csv(data_dir + '/SPY.csv')
spy_df.index = pd.to_datetime(spy_df['Date'])

##------Explore the data with some EDA----
#lng_df = lng_df.loc[:, ['Adj_Close', 'Adj_Volume']]
#spy_df = spy_df.loc[:, ['Adj_Close', 'Adj_Volume']]
#
#print(lng_df.head())  # examine the DataFrames
#print(spy_df.head())  # examine the SPY DataFrame
#
## Plot the Adj_Close columns for SPY and LNG
#spy_df['Adj_Close'].plot(label='SPY', legend=True)
#lng_df['Adj_Close'].plot(label='LNG', legend=True, secondary_y=True)
#plt.show()  # show the plot
#plt.clf()  # clear the plot space
#
## Histogram of the daily price change percent of Adj_Close for LNG
#lng_df['Adj_Close'].pct_change(1).plot.hist(bins=50)
#plt.xlabel('adjusted close 1-day percent change')
#plt.show()

#------Correlation--------
#used dataset
lng_df = lng_df.loc[:, ['Adj_Close', 'Adj_Volume']]
# Create 5-day % changes of Adj_Close for the current day, and 5 days in the future
lng_df['5d_future_close'] = lng_df['Adj_Close'].shift(-5)
lng_df['5d_close_future_pct'] = lng_df['5d_future_close'].pct_change(5)
lng_df['5d_close_pct'] = lng_df['Adj_Close'].pct_change(5)
# Calculate the correlation matrix between the 5d close pecentage changes (current and future)
corr = lng_df[['5d_close_pct', '5d_close_future_pct']].corr()
print(corr)
# Scatter the current 5-day percent change vs the future 5-day percent change
plt.scatter(lng_df['5d_close_pct'], lng_df['5d_close_future_pct'])
plt.xlabel('5d_close_pct')
plt.ylabel('5d_close_future_pct')
plt.show()

#------Create moving average and RSI features--------
feature_names = ['5d_close_pct']  # a list of the feature names for later
# Create moving averages and rsi for timeperiods of 14, 30, 50, and 200
for n in [14, 30, 50, 200]:
    # Create the moving average indicator and divide by Adj_Close
    lng_df['ma' + str(n)] = talib.SMA(lng_df['Adj_Close'].values,
                              timeperiod=n) / lng_df['Adj_Close']
    # Create the RSI indicator
    lng_df['rsi' + str(n)] = talib.RSI(lng_df['Adj_Close'].values, timeperiod=n)  
    # Add rsi and moving average to the feature name list
    feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]
    
print(feature_names)

#------Create features and targets-----
# Drop all na values
lng_df = lng_df.dropna()
# Create features and targets
# use feature_names for features; 5d_close_future_pct for targets
features = lng_df[feature_names]
targets = lng_df['5d_close_future_pct']
# Create DataFrame from target column and feature columns
feat_targ_df = lng_df[['5d_close_future_pct'] + feature_names]
# Calculate correlation matrix
corr = feat_targ_df.corr()
print(corr)

#-----Check the correlations-----
# Plot heatmap of correlation matrix
sns.heatmap(corr, annot=True)
plt.yticks(rotation=0); plt.xticks(rotation=90)  # fix ticklabel directions
plt.tight_layout()  # fits plot area to the plot, "tightly"
plt.show()  # show the plot
plt.clf()  # clear the plot area
# Create a scatter plot of the most highly correlated variable with the target
plt.scatter(lng_df['rsi200'], lng_df['5d_close_future_pct'])
plt.show()

#------Create train and test features--
# Add a constant to the features
linear_features = sm.add_constant(features)
# Create a size for the training set that is 85% of the total number of samples
train_size = int(0.85 * targets.shape[0])
train_features = linear_features[:train_size]
train_targets = targets[:train_size]
test_features = linear_features[train_size:]
test_targets = targets[train_size:]
print(linear_features.shape, train_features.shape, test_features.shape)

#----Fit a linear model---
# Create the linear model and complete the least squares fit
model = sm.OLS(train_targets, train_features)
results = model.fit()
print(results.summary())
# examine p-values
# Features with p <= 0.05 are typically considered significantly different from 0
print(results.pvalues)
# Make predictions from our model for train and test sets
train_predictions = results.predict(train_features)
test_predictions = results.predict(test_features)

#----------Evaluate our linear model's results ----
# Scatter the predictions vs the targets with 80% transparency
plt.scatter(train_predictions, train_targets, alpha=0.2, color='b', \
            label='train')
plt.scatter(test_predictions, test_predictions, alpha=0.2, color='r', \
            label='test')
# Plot the perfect prediction line
xmin, xmax = plt.xlim()
plt.plot(np.arange(xmin, xmax, 0.01), np.arange(xmin, xmax, 0.01), c='k')
plt.xlabel('predictions')
plt.ylabel('actual')
plt.legend()
plt.show()

#----------Feature engineering from volume---
"""
results analysis: 
    We can see the moving average of volume changes has a much smaller range 
    than the raw data.
"""
# Create 2 new volume features, 1-day % change and 5-day SMA of the % change
new_features = ['Adj_Volume_1d_change', 'Adj_Volume_1d_change_SMA']
#add new features
lng_df['Adj_Volume_1d_change'] = lng_df['Adj_Volume'].pct_change(1)
lng_df['Adj_Volume_1d_change_SMA'] = talib.SMA(lng_df['Adj_Volume_1d_change'].values, \
      timeperiod=5)
# Plot histogram of volume % change data
lng_df[new_features].plot(kind='hist', sharex=False, bins=50)
plt.show()

#----------Create day-of-week features---
# Use pandas' get_dummies function to get dummies for day of the week
days_of_week = pd.get_dummies(lng_df.index.dayofweek, drop_first=True, \
                              prefix='weekday')
# Set the index as the original dataframe index for merging
days_of_week.index = lng_df.index
# Join the dataframe with the days of week dataframe
lng_df = pd.concat([lng_df, days_of_week], axis=1)
lng_df.dropna(inplace=True) # drop missing values in-place
print(lng_df.head())

#----------Examine correlations of the new features----
"""
results analysis:
    Even though the correlations are weak, they may improve our predictions 
    via interactions with other features.
"""
# Add the weekday labels to the new_features list
new_features.extend(['weekday_' + str(i) for i in range(1,5)])
#add new features in feature_names
feature_names.extend(new_features)
# Plot the correlations between the new features and the targets
sns.heatmap(lng_df[new_features + ['5d_close_future_pct']].corr(), annot=True)
plt.yticks(rotation=0) # ensure y-axis ticklabels are horizontal
plt.xticks(rotation=90) # ensure x-axis ticklabels are vertical
plt.tight_layout()
plt.show()

#-----Fit a decision tree---
"""
Results Analysis:
    A perfect fit! ...on the training data. OVERFITTING PROBLEM
    
    Score or R-squared:
        - 1 means perfect fit
        - 0 means bad
        - negative means terrible result that the case in this model
"""
# Create a size for the training set that is 85% of the total number of samples
targets = lng_df['5d_close_future_pct']
train_size = int(0.85 * targets.shape[0])
train_features = lng_df[:train_size][feature_names]
train_targets = targets[:train_size]
test_features = lng_df[train_size:][feature_names]
test_targets = targets[train_size:]
print(lng_df.shape, train_features.shape, test_features.shape)
# Create a decision tree regression model with default arguments
decision_tree = DecisionTreeRegressor()
# Fit the model to the training features and targets
decision_tree.fit(train_features, train_targets)
# Check the score on train and test
print(decision_tree.score(train_features, train_targets))
print(decision_tree.score(test_features, test_targets))

#----Try with different max depths to reduce the overfitting problem----
"""
Results Analysis:
    we got the highest R-squared score for max_depth = 5
    The predictions group into lines because our depth is limited.
"""
# Loop through a few different max depths and check the performance
# i.e. R-squared scores.
for d in [3,5,10]:
    # Create the tree and fit it
    decision_tree = DecisionTreeRegressor(max_depth=d)
    decision_tree.fit(train_features, train_targets)
    # Print out the scores on train and test
    print('max_depth=', str(d))
    print(decision_tree.score(train_features, train_targets))
    print(decision_tree.score(test_features, test_targets), '\n')

# Use the best max_depth of 5 from last exercise to fit a decision tree
decision_tree = DecisionTreeRegressor(max_depth=5)
decision_tree.fit(train_features, train_targets)
# Predict values for train and test
train_predictions = decision_tree.predict(train_features)
test_predictions = decision_tree.predict(test_features)
# Scatter the predictions vs actual values
plt.scatter(train_predictions, train_targets, label='train')
plt.scatter(test_predictions, test_targets, label='test')
plt.legend()
plt.show()

#----Fit a random forest-----
"""
Results Analysis:
    Our test score (R^2) isn't great, but it's greater than 0!
"""
# Create the random forest model and fit to the training data
rfr = RandomForestRegressor(n_estimators=200)
rfr.fit(train_features, train_targets)
# Look at the R^2 scores on train and test
print(rfr.score(train_features, train_targets))
print(rfr.score(test_features, test_targets))

#----Tune random forest hyperparameters---
"""
Results Analysis:
    We can see our train predictions are good, but test predictions 
    (generalization) are not great.
"""
# Create a dictionary of hyperparameters to search
grid = {'n_estimators': [200], 'max_depth': [5], 'max_features': [4,8], \
        'random_state': [42]}
test_scores = []
# Loop through the parameter grid, set the hyperparameters, and save the scores
for g in ParameterGrid(grid):
    rfr.set_params(**g)  # ** is "unpacking" the dictionary
    rfr.fit(train_features, train_targets)
    test_scores.append(rfr.score(test_features, test_targets))

# Find best hyperparameters from the test score and print
best_idx = np.argmax(test_scores)
print(test_scores[best_idx], ParameterGrid(grid)[best_idx])

#------Evaluate performance-----------
# Use the best hyperparameters from before to fit a random forest model
rfr = RandomForestRegressor(n_estimators=200, max_depth=5, max_features=4, \
                            random_state=42)
rfr.fit(train_features, train_targets)
# Make predictions with our model
train_predictions = rfr.predict(train_features)
test_predictions = rfr.predict(test_features)
# Create a scatter plot with train and test actual vs predictions
plt.scatter(train_targets, train_predictions, label='train')
plt.scatter(test_targets, test_predictions, label='test')
plt.legend()
plt.show()

#--------Random forest feature importances------
"""
Results Analysis:
    Unsurprisingly, it looks like the days of the week should be thrown out.
"""
# Get feature importances from our random forest model
importances = rfr.feature_importances_
# Get the index of importances from greatest importance to least
sorted_index = np.argsort(importances)[::-1] # [::-1] reverts the indexes
x = range(len(importances))
# Create tick labels 
labels = np.array(feature_names)[sorted_index]
plt.bar(x, importances[sorted_index], tick_label=labels)
# Rotate tick labels to vertical
plt.xticks(rotation=90)
plt.show()

#------A gradient boosting model-------
"""
Results Analysis::
    In this case the gradient boosting model isn't that much better than a 
    random forest, but you know what they say -- no free lunch!
"""
# Create GB model -- hyperparameters have already been searched for you
gbr = GradientBoostingRegressor(max_features=4,
                                learning_rate=0.01,
                                n_estimators=200,
                                subsample=0.6,
                                random_state=42)
gbr.fit(train_features, train_targets)

print(gbr.score(train_features, train_targets))
print(gbr.score(test_features, test_targets))

#----------Gradient boosting feature importances--------------
"""
Results Analysis:
    Notice the feature importances are not exactly the same as the random 
    forest model's...but they're close.
"""
# Extract feature importances from the fitted gradient boosting model
feature_importances = gbr.feature_importances_
# Get the indices of the largest to smallest feature importances
sorted_index = np.argsort(feature_importances)[::-1]
x = range(len(feature_importances))
# Create tick labels 
labels = np.array(feature_names)[sorted_index]
plt.bar(x, feature_importances[sorted_index], tick_label=labels)
# Set the tick lables to be the feature names, according to the sorted feature_idx
plt.xticks(rotation=90)
plt.show()

#-----Standardizing data----------
# Remove unimportant features (weekdays)
train_features = train_features.iloc[:, :-4]
test_features = test_features.iloc[:, :-4]

# Standardize the train and test features
scaled_train_features = scale(train_features)
scaled_test_features = scale(test_features)

# Plot histograms of the 14-day SMA RSI before and after scaling
f, ax = plt.subplots(nrows=2, ncols=1)
train_features.iloc[:, 2].hist(ax=ax[0])
ax[1].hist(scaled_train_features[:, 2])
plt.show()

#-----Optimize n_neighbors----------
"""
Results Analysis:
    See how n is the best number of neighbors based on the test scores?
"""
for n in range(2,13):
    # Create and fit the KNN model
    knn = KNeighborsRegressor(n_neighbors=n)
    
    # Fit the model to the training data
    knn.fit(train_features, train_targets)
    
    # Print number of neighbors and the score to find the best value of n
    print("n_neighbors =", n)
    print('train, test scores')
    print(knn.score(scaled_train_features, train_targets))
    print(knn.score(scaled_test_features, test_targets))
    print()  # prints a blank line
    
#-------Evaluate KNN performance-----------
"""
Results Analysis:
    the model is doing OK!
"""
# Create the model with the best-performing n_neighbors of 8
knn = KNeighborsRegressor(n_neighbors=8)

# Fit the model
knn.fit(scaled_train_features, train_targets)

# Get predictions for train and test sets
train_predictions = knn.predict(scaled_train_features)
test_predictions = knn.predict(scaled_test_features)

# Plot the actual vs predicted values
plt.scatter(train_predictions, train_targets, label='train')
plt.scatter(test_predictions, test_targets, label='test')
plt.legend()
plt.show()

#----Build and fit a simple neural net----
"""
Results Analysis:
    Now we need to check that our training loss has flattened out and the net 
    is sufficiently trained.
"""
# Create the model
# Create the model
model_1 = Sequential()
model_1.add(Dense(100, input_dim=scaled_train_features.shape[1], activation='relu'))
model_1.add(Dense(20, activation='relu'))
model_1.add(Dense(1, activation='linear'))

# Fit the model
model_1.compile(optimizer='adam', loss='mse')
history = model_1.fit(scaled_train_features, train_targets, epochs=25)

#---------Plot losses---------
"""
Results Analysis:
    We can see our loss has flattened out, so we're good!
"""
# Plot the losses from the fit
plt.plot(history.history['loss'])
# Use the last loss as the title
plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
plt.show()

#--------Measure performance-----------
"""
Results Analysis:
    It doesn't look too much different from our other models at this point.
"""
# Calculate R^2 score
train_preds = model_1.predict(scaled_train_features)
test_preds = model_1.predict(scaled_test_features)
print(r2_score(train_targets, train_preds))
print(r2_score(test_targets, test_preds))

# Plot predictions vs actual
plt.scatter(train_preds, train_targets, label='train')
plt.scatter(test_preds, test_targets, label='test')
plt.legend()
plt.show()

#-----Fit neural net with custom loss function------
"""
Results Analysis:
    Notice how the train set actual vs predictions shape has changed to be a 
    bow-tie.
"""
keras.losses.sign_penalty = sign_penalty  # enable use of loss with keras
print(keras.losses.sign_penalty)
# Create the model
model_2 = Sequential()
model_2.add(Dense(100, input_dim=scaled_train_features.shape[1], activation='relu'))
model_2.add(Dense(20, activation='relu'))
model_2.add(Dense(1, activation='linear'))

# Fit the model with our custom 'sign_penalty' loss function
model_2.compile(optimizer='adam', loss=sign_penalty)
history = model_2.fit(scaled_train_features, train_targets, epochs=25)
plt.plot(history.history['loss'])
plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
plt.show()

#----------Combatting overfitting with dropout---------
"""
Results Analysis:
    Dropout helps the model generalized a bit better to unseen data.
"""
# Create model with dropout
model_3 = Sequential()
model_3.add(Dense(100, input_dim=scaled_train_features.shape[1], activation='relu'))
model_3.add(Dropout(0.2))
model_3.add(Dense(20, activation='relu'))
model_3.add(Dense(1, activation='linear'))

# Fit model with mean squared error loss function
model_3.compile(optimizer='adam', loss='mse')
history = model_3.fit(scaled_train_features, train_targets, epochs=25)
plt.plot(history.history['loss'])
plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
plt.show()

#------Ensembling models (enseble of average scores)---------
"""
Results Analysis:
    Our R^2 values are around the average of the 3 models we ensembled. Notice 
    the plot also looks like the bow-tie shape has been softened a bit.
"""
# Make predictions from the 3 neural net models
train_pred1 = model_1.predict(scaled_train_features)
test_pred1 = model_1.predict(scaled_test_features)

train_pred2 = model_2.predict(scaled_train_features)
test_pred2 = model_2.predict(scaled_test_features)

train_pred3 = model_3.predict(scaled_train_features)
test_pred3 = model_3.predict(scaled_test_features)

# Horizontally stack predictions and take the average across rows
train_preds = np.mean(np.hstack((train_pred1, train_pred2, train_pred3)), axis=1)
test_preds = np.mean(np.hstack((test_pred1, test_pred2, test_pred3)), axis=1)
print(test_preds[-5:])

#See how the ensemble performed
# Evaluate the R^2 scores
print(r2_score(train_targets, train_preds))
print(r2_score(test_targets, test_preds))

# Scatter the predictions vs actual -- this one is interesting!
plt.scatter(train_preds, train_targets, label='train')
plt.scatter(test_preds, test_targets, label='test')
plt.legend(); plt.show()

#-----Modern Portfolio Theory (MPT)-------- 

#preapre three datasets: lng_df, spy_df, and smlv_df
# read SMLV data from Excel and prepare smlv_df DataFrame
smlv_df = pd.ExcelFile(data_dir + '/SMLV.xlsx')
smlv_df = smlv_df.parse('Sheet1')
smlv_list = []
for i in range(len(smlv_df)):
    val = smlv_df.iloc[i]
    smlv_list.append(str(val[0]).split('  ')) 

smlv_df = pd.DataFrame(smlv_list)
smlv_df.columns = ['Date', 'SMLV']
smlv_df.index = pd.to_datetime(smlv_df['Date'])
smlv_df = smlv_df.drop('Date', axis=1)
smlv_df['SMLV'] = smlv_df['SMLV'].astype(float)
## preapre lng_df
#lng_df = lng_df.loc[:, ['Adj_Close']]
#lng_df.columns = ['LNG']
# preapre spy_df
spy_df = pd.read_csv(data_dir + '/SPY.csv')
spy_df.index = pd.to_datetime(spy_df['Date'])
spy_df = spy_df.loc[:, ['Adj_Close']]
spy_df.columns = ['SPY']

# Join 3 stock dataframes together
full_df = pd.concat([lng_df, spy_df, smlv_df], axis=1).dropna()
# Resample the full dataframe to monthly timeframe
monthly_df = full_df.resample('BMS').first()
# Calculate daily returns of stocks
returns_daily = full_df.pct_change().dropna()
# Calculate monthly returns of the stocks
returns_monthly = monthly_df.pct_change().dropna()
print(returns_monthly.tail())

# Daily covariance of stocks (for each monthly period)
covariances = {}
rtd_idx = returns_daily.index
for i in returns_monthly.index:    
    # Mask daily returns for each month and year, and calculate covariance
    mask = (rtd_idx.month == i.month) & (rtd_idx.year == i.year)
    # Use the mask to get daily returns for the current month and year of monthy returns index
    covariances[i] = returns_daily[mask].cov()

print(covariances[i])

#Calculate portfolios
portfolio_returns, portfolio_volatility, portfolio_weights = {}, {}, {}
# Get portfolio performances at each month
for date in sorted(covariances.keys()):
    cov = covariances[date]
    for portfolio in range(10):
        weights = np.random.random(3)
        weights /= np.sum(weights) # /= divides weights by their sum to normalize
        returns = np.dot(weights.T, returns_monthly.loc[date])
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        portfolio_returns.setdefault(date, []).append(returns)
        portfolio_volatility.setdefault(date, []).append(volatility)
        portfolio_weights.setdefault(date, []).append(weights)
        
print(portfolio_weights[date][0])

#Plot efficient frontier
"""
Results Analysis:
    Often the efficient frontier will be a bullet shape, but if the returns 
    are all positive then it may look like this.
"""
# Get latest date of available data
date = sorted(covariances.keys())[-1]  
# Plot efficient frontier
# warning: this can take at least 10s for the plot to execute...
plt.scatter(x=portfolio_volatility[date], y=portfolio_returns[date],  alpha=0.1)
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.show()

#Get best Sharpe ratios for each month or (BMS)
"""
Results Analysis:
    We've got our best Sharpe ratios, which we'll use to create targets for 
    machine learning.
"""
# Empty dictionaries for sharpe ratios and best sharpe indexes by date
sharpe_ratio, max_sharpe_idxs = {}, {}
# Loop through dates and get sharpe ratio for each portfolio
for date in portfolio_returns.keys():
    for i, ret in enumerate(portfolio_returns[date]):
        # Divide returns by the volatility for the date and index, i
        sharpe_ratio.setdefault(date, []).append(ret / portfolio_volatility[date][i])
    # Get the index of the best sharpe ratio for each date
    max_sharpe_idxs[date] = np.argmax(sharpe_ratio[date])

print(portfolio_returns[date][max_sharpe_idxs[date]])

#Calculate EWMAs
# Calculate exponentially-weighted moving average of daily returns
ewma_daily = returns_daily.ewm(span=30).mean()
# Resample daily returns to first business day of the month with average for that month
ewma_monthly = ewma_daily.resample('BMS').first()
# Shift ewma for the month by 1 month forward so we can use it as a feature for future predictions 
ewma_monthly = ewma_monthly.shift(1).dropna()
print(ewma_monthly.iloc[-1])

#Make features and targets
targets, features = [], []
# Create features from price history and targets as ideal portfolio
for date, ewma in ewma_monthly.iterrows():
    # Get the index of the best sharpe ratio
    best_idx = max_sharpe_idxs[date]
    targets.append(portfolio_weights[date][best_idx])
    features.append(ewma)  # add ewma to features
targets = np.array(targets)
features = np.array(features)
print(targets[-5:])

#Plot efficient frontier with best Sharpe ratio
"""
Results Analysis:
    The best portfolio according to Sharpe is usually somewhere in that area 
    where the orange x is.
"""
# Get most recent (current) returns and volatility
date = sorted(covariances.keys())[-1]
cur_returns = portfolio_returns[date]
cur_volatility = portfolio_volatility[date]
# Plot efficient frontier with sharpe as point
plt.scatter(x=cur_volatility, y=cur_returns, alpha=0.1, color='blue')
best_idx = max_sharpe_idxs[date]
# Place an orange "X" on the point with the best Sharpe ratio
plt.scatter(x=cur_volatility[best_idx], y=cur_returns[best_idx], marker='x', color='orange')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.show()

#Make predictions with a random forest
"""
Results Analysis:
    The test score is not so good, but it'll work out OK in this case.
"""
# Make train and test features
train_size = int(0.85 * features.shape[0])
train_features = features[:train_size]
test_features = features[train_size:]
train_targets = targets[:train_size]
test_targets = targets[train_size:]

# Fit the model and check scores on train and test
rfr = RandomForestRegressor(n_estimators=300, random_state=42)
rfr.fit(train_features, train_targets)
print(rfr.score(train_features, train_targets))
print(rfr.score(test_features, test_targets))

#Get predictions and first evaluation
"""
Results Analysis:
    We're doing a little better than SPY sometimes, and other times not. 
    Let's see how it adds up...
"""
# Get predictions from model on train and test
train_predictions = rfr.predict(train_features)
test_predictions = rfr.predict(test_features)
# Calculate and plot returns from our RF predictions and the SPY returns
test_returns = np.sum(returns_monthly.iloc[train_size:] * test_predictions, axis=1)
plt.plot(test_returns, label='algo')
plt.plot(returns_monthly['SPY'].iloc[train_size:], label='SPY')
plt.legend()
plt.show()

#Evaluate returns
"""
Results Analysis:
    Our predictions slightly beat the SPY!
"""
# Calculate the effect of our portfolio selection on a hypothetical $1k investment
cash = 1000
algo_cash, spy_cash = [cash], [cash]  # set equal starting cash amounts
for r in test_returns:
    cash *= 1 + r
    algo_cash.append(cash)
# Calculate performance for SPY
cash = 1000  # reset cash amount
for r in returns_monthly['SPY'].iloc[train_size:]:
    cash *= 1 + r
    spy_cash.append(cash)
print('algo returns:', (algo_cash[-1] - algo_cash[0]) / algo_cash[0])
print('SPY returns:', (spy_cash[-1] - spy_cash[0]) / spy_cash[0])

#Plot returns
# Plot the algo_cash and spy_cash to compare overall returns
plt.plot(algo_cash, label='algo')
plt.plot(spy_cash, label='SPY')
plt.legend()  # show the legend
plt.show()

# =================End of analyzing AMD dataset ===============================






















