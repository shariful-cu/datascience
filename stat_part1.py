#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 05:57:30 2019

@author: Shariful
"""
#---- import necessary modules -----
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import datetime


#---- global values -----
data_dir = '/Users/Shariful/Documents/GitHubRepo/Datasets'
sns.set()
# setting font type and style
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

#---- utility functions -----
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / float(n)
    return x, y

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x,y)
    # Return entry [0,1]
    return corr_mat[0,1]

def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0
    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()
        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success += 1

    return n_success

def successive_poisson(tau1, tau2, size=1):
    """Compute time for arrival of 2 successive Poisson processes."""
    # Draw samples out of first exponential distribution: t1
    t1 = np.random.exponential(tau1, size=size)

    # Draw samples out of second exponential distribution: t2
    t2 = np.random.exponential(tau2, size=size)

    return t1 + t2

#======================================= Analysis of Iris dataset =============

##---- loading data -----
#df_iris = load_iris()
#data_iris = df_iris.data
#iris_lab = df_iris.target
#iris_dataframe = pd.DataFrame(data_iris)
#iris_dataframe.columns = df_iris.feature_names
#iris_lab_cat = list(iris_lab)
#iris_lab_cat = [df_iris.target_names[0] if x == 0 else x for x in iris_lab_cat]
#iris_lab_cat = [df_iris.target_names[1] if x == 1 else x for x in iris_lab_cat]
#iris_lab_cat = [df_iris.target_names[2] if x == 2 else x for x in iris_lab_cat]
#iris_dataframe['species'] = iris_lab_cat
#
##---- rough ---
##or df_iris.target_names[1] if x == 1 or
##                             df_iris.target_names[2] if x == 2
#
##---Histogram----
#versicolor_petal_length = data_iris[iris_lab == 1, 2]
#n_bins = int(np.sqrt(len(versicolor_petal_length)))
#plt.hist(versicolor_petal_length, bins=n_bins)
#plt.xlabel('petal length (cm)')
#plt.ylabel('count')
#plt.show()

##---Swarm plot----
#sns.swarmplot(x='species', y='petal length (cm)', data=iris_dataframe)
#plt.xlabel('species')
#plt.ylabel('petal length (cm)')
#plt.show()

##---box plot---
#sns.boxplot(x='species', y='petal length (cm)', data=iris_dataframe)
#plt.xlabel('species')
#plt.ylabel('petal length (cm)')
#plt.show()

####---ECDF plot----
#setosa_petal_length = data_iris[iris_lab == 0, 2]
#versicolor_petal_length = data_iris[iris_lab == 1, 2]
#virginica_petal_length = data_iris[iris_lab == 2, 2]
#
## Compute ECDFs
#x_set, y_set = ecdf(setosa_petal_length)
#x_vers, y_vers = ecdf(versicolor_petal_length)
#x_virg, y_virg = ecdf(virginica_petal_length)
#
## Plot all ECDFs on the same plot
#plt.plot(x_set, y_set, marker='.', linestyle='none')
#plt.plot(x_vers, y_vers, marker='.', linestyle='none')
#plt.plot(x_virg, y_virg, marker='.', linestyle='none')
#
## Annotate the plot
#plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
#plt.xlabel('petal length (cm)')
#plt.ylabel('ECDF')
#
## Display the plot
#plt.show()

###---Percentile vs Ecdf----
#versicolor_petal_length = data_iris[iris_lab == 1, 2]
## Specify array of percentiles: percentiles
#percentiles = np.array([2.5, 25, 50, 75, 97.5])
## Compute percentiles: ptiles_vers
#ptiles_vers = np.percentile(versicolor_petal_length, percentiles)
## Print the result
#print(ptiles_vers)
##compute ecdf
#x_vers, y_vers = ecdf(versicolor_petal_length)
##plot ecdf
#plt.plot(x_vers, y_vers, marker='.', linestyle='none')
##plot percentiles
#plt.plot(ptiles_vers, percentiles/100, marker='D', color='red', linestyle='none')
#
#plt.xlabel('petal length (cm)')
#plt.ylabel('ECDF')
#plt.show()

##---Variance and Standard Deviation---
#versicolor_petal_length = data_iris[iris_lab == 1, 2]
#print(np.var(versicolor_petal_length))
#print(np.std(versicolor_petal_length))

###---Covariance and Pearson Correlation-----
#versicolor_petal_length = data_iris[iris_lab == 1, 2]
#versicolor_petal_width = data_iris[iris_lab == 1, 3]
##scatter plot
#plt.plot(versicolor_petal_length, versicolor_petal_width, marker='.', linestyle='none')
#plt.xlabel('versicolor petal length (cm)')
#plt.ylabel('versicolor petal width (cm)')
#plt.show()
##Covariance
#covariance_matrix = np.cov(versicolor_petal_length, versicolor_petal_width)
#print('\nCovariance between versicolor\'s petal length and pletal width is: ' \
#      + str(covariance_matrix[0,1]))
##Pearson Correlation
#r = pearson_r(versicolor_petal_length, versicolor_petal_width)
##print('\nPearson Correlation is: ' + str(r))


#====== End of Analysis of Iris dataset =======================================


#======== Analysis of  2008_swing_states eLection dataset =====================

##--loading data---
#df_swing = pd.read_csv(data_dir + '/2008_swing_states.csv')
##df_swing[['state', 'county', 'dem_share']].head()

##---Histogram----
#plt.hist(df_swing['dem_share'])
#plt.xlabel('% of vote for Obama')
#plt.ylabel('number of counties')
#plt.show()

##---Swarm plot----
#sns.swarmplot(x='state', y='dem_share', data=df_swing)
#plt.xlabel('state')
#plt.ylabel('% of votes for Obama')
#plt.show()

###---ECDF plot----
#x_PA, y_PA = ecdf(df_swing[df_swing['state']=='PA']['dem_share'])
#plt.plot(x_PA, y_PA, marker='.', linestyle='none')
#
#x_OH, y_OH = ecdf(df_swing[df_swing['state']=='OH']['dem_share'])
#plt.plot(x_OH, y_OH, marker='.', linestyle='none')
#
#x_FL, y_FL = ecdf(df_swing[df_swing['state']=='FL']['dem_share'])
#plt.plot(x_FL, y_FL, marker='.', linestyle='none')
#
#plt.xlabel('% of votes for Obama')
#plt.ylabel('ECDF')
#plt.margins(0.02)
#plt.show()

##---percentile----
#print(np.percentile(df_swing['dem_share'], [25, 50, 75]))

###---scatter plot between total votes vs % of votes---
#plt.plot(df_swing['total_votes']/1000, df_swing['dem_share'], marker='.', \
#         linestyle='none')
##plotting means of x-axis and y-axis
#mean_x = np.mean(df_swing['total_votes']/1000)
#mean_y = np.mean(df_swing['dem_share'])
#plt.plot(np.array([mean_x,mean_x]), np.array([0,max(df_swing['dem_share'])]))
#plt.plot(np.array([0,max(df_swing['total_votes']/1000)]), np.array([mean_y,mean_y]))
##printing a message for mean of total votes
#plt.text(100, 10, 'mean of total votes', fontdict=font)
##printing a message for mean of % of votes
#plt.text(400, 37, 'mean of % of votes', fontdict=font)
##labeling axis
#plt.xlabel('total votes (thousands)')
#plt.ylabel('% of votes for Obama')
#plt.show()


##====== End of Analysis of 2008_swing_states  ELection dataset ===============




#======== Analysis of  2008_all_states eLection dataset =======================

##----Loading data-----
#df_swing = pd.read_csv(data_dir + '/2008_all_states.csv')
##df_swing.head()

##---box plot---
#sns.boxplot(x='east_west', y='dem_share', data=df_swing)
#plt.xlabel('region')
#plt.ylabel('% of votes for Obama')
#plt.show()


#====== End of Analysis of 2008_all_states ELection dataset ===================



#======== probabilistic inference =============================================

##-----Bernoulli trials, 100 mortgage loans----
## Seed random number generator
#np.random.seed(42)
## Initialize the number of defaults: n_defaults
#n_defaults = np.empty(1000)
## Compute the number of defaults
#for i in range(1000):
#    n_defaults[i] = perform_bernoulli_trials(100,0.05)
## Plot the histogram with default number of bins; label your axes
#_ = plt.hist(n_defaults, normed=True)
#_ = plt.xlabel('number of defaults out of 100 loans')
#_ = plt.ylabel('probability')
## Show the plot
#plt.show()
## Compute ECDF: x, y
#x,y = ecdf(n_defaults)
## Plot the ECDF with labeled axes
#plt.plot(x, y, marker='.', linestyle='none')
#plt.xlabel('number of defaults')
#plt.ylabel('ECDF')
## Show the plot
#plt.show()
## Compute the number of 100-loan simulations with 10 or more defaults: n_lose_money
#n_lose_money = np.sum(n_defaults >= 10)
## Compute and print probability of losing money
#print('Probability of losing money =', n_lose_money / len(n_defaults))

###-----Binomial distribution, 100 mortgage loans----
#np.random.seed(42)
#n_defaults = np.random.binomial(100, 0.05, size=10000)
#x, y = ecdf(n_defaults)
#plt.plot(x, y, marker='.', linestyle='none')
#plt.xlabel('number of defaults')
#plt.ylabel('ECDF')
#plt.show()
#
## Compute bin edges: bins
#bins = np.arange(0, max(n_defaults) + 1.5) - 0.5
## Generate histogram
#plt.hist(n_defaults, bins=bins, normed=True)
## Label axes
#plt.xlabel('number of defaults')
#plt.ylabel('PMF')
## Show the plot
#plt.show()

###-----Poisson distribution, number of hits on a webpage----
#samples = np.random.poisson(6, size=10000)
#x, y = ecdf(samples)
#plt.plot(x, y, marker='.', linestyle='none')
#plt.xlabel('number of hits')
#plt.ylabel('ECDF')
#plt.show()
#
###-----Relationship between Poisson and Binomial distribution----
## Draw 10,000 samples out of Poisson distribution: samples_poisson
#samples_poisson = np.random.poisson(10, size=10000)
## Print the mean and standard deviation
#print('Poisson:     ', np.mean(samples_poisson),
#                       np.std(samples_poisson))
## Specify values of n and p to consider for Binomial: n, p
#n = [20, 100, 1000]
#p = [0.5, 0.1, 0.01]
## Draw 10,000 samples for each n,p pair: samples_binomial
#for i in range(3):
#    samples_binomial = np.random.binomial(n[i], p[i], size=10000)
#    # Print results
#    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
#                                 np.std(samples_binomial))

##-----find probablity of a rare event (e.g., probability of most no-hitters---
##------in  a bassball game--
"""
Was 2015 anomalous?
1990 and 2015 featured the most no-hitters of any season of baseball 
(there were seven). Given that there are on average 251/115 no-hitters per 
season, what is the probability of having seven or more in a season?
"""    
## Draw 10,000 samples out of Poisson distribution: n_nohitters
#n_nohitters = np.random.poisson(251/115, 10000)
## Compute number of samples that are seven or greater: n_large
#n_large = np.sum(n_nohitters >= 7)
## Compute probability of getting seven or more: p_large
#p_large = n_large/10000
## Print the result
#print('Probability of seven or more no-hitters:', p_large)
    
##---The Normal PDF-----
## Draw 100000 samples from Normal distribution with meand 20  
## and stds (1, 3, 10) of interest: samples_std1, samples_std3, samples_std10
#samples_std1 = np.random.normal(20, 1, size=100000)
#samples_std3 = np.random.normal(20, 3, size=100000)
#samples_std10 = np.random.normal(20, 10, size=100000)
## Make histograms
#plt.hist(samples_std1, bins=100, normed=True, histtype='step')
#plt.hist(samples_std3, bins=100, normed=True, histtype='step')
#plt.hist(samples_std10, bins=100, normed=True, histtype='step')
## Make a legend, set limits and show plot
#plt.legend(('std = 1', 'std = 3', 'std = 10'))
#plt.ylim(-0.01, 0.42)
#plt.show()
## Generate CDFs
#x_std1, y_std1 = ecdf(samples_std1)
#x_std3, y_std3 = ecdf(samples_std3)
#x_std10, y_std10 = ecdf(samples_std10)
## Plot CDFs
#plt.plot(x_std1, y_std1, marker='.', linestyle='none')
#plt.plot(x_std3, y_std3, marker='.', linestyle='none')
#plt.plot(x_std10, y_std10, marker='.', linestyle='none')
## Make a legend and show the plot
#plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
#plt.show()

#======== End of probabilistic inference ======================================


##======== Normal Distribution using Michel's Speed of Light dataset ==========
#
##--loading data---
#df_speed_of_light = pd.read_csv(data_dir + '/michelson_speed_of_light.csv')
##df_speed_of_light.head()
#
##--comparing ecdf between theoritical normal distribution vs actual one---
#michelson_speed_of_light = df_speed_of_light['velocity of light in air (km/s)']
#mean = np.mean(michelson_speed_of_light)
#std = np.std(michelson_speed_of_light)
#samples = np.random.normal(mean, std, size=10000)
#
#x,y = ecdf(michelson_speed_of_light)
#x_theor, y_theor = ecdf(samples)
#
#plt.plot(x_theor, y_theor, marker='.', linestyle='none')
#plt.plot(x, y, marker='.', linestyle='none')
#
#plt.xlabel('speed of light (km/s)')
#plt.ylabel('ECDF')
#plt.legend(('theoritical normal distribution', 'original speed of light'), \
#           loc='lower right')
#plt.show()
#
##======== End of Normal Distribution using Michel's Speed of Light dataset====
    

##======== Normal Distribution using Belmont dataset =========================
    
###--loading data---
#df_belmont = pd.read_csv(data_dir + '/belmont.csv')
##df_belmont.head()
#
##convert df_belmont['Time'] into minutes
#HM = [60, 1]
#belmont_with_outliers = []
#idx=0
#for t in df_belmont['Time']:
#    time_m = sum(a * b for a, b in zip(HM, map(float, t.split(":"))))
#    belmont_with_outliers.append(time_m)
#
## draw boxplot and detect outliers (two points)
#belmont_with_outliers = np.array(belmont_with_outliers)
#sns.boxplot(data=belmont_with_outliers)
#plt.show()
##remove two outliers points
#belmont_no_outliers = belmont_with_outliers[belmont_with_outliers > 144]
#belmont_no_outliers = belmont_no_outliers[belmont_no_outliers < 154]
#
##--comparing ecdf between theoritical normal distribution vs actual one---
#mean = np.mean(belmont_no_outliers)
#std = np.std(belmont_no_outliers)
#samples = np.random.normal(mean, std, size=10000)
#x,y = ecdf(belmont_no_outliers)
#x_theor, y_theor = ecdf(samples)
#plt.plot(x_theor, y_theor)
#plt.plot(x, y, marker='.', linestyle='none')
#plt.xlabel('Belmont winning time (sec.)')
#plt.ylabel('CDF')
#plt.legend(('theoritical normal distribution', 'Belmont winning time (sec.)'), \
#           loc='lower right')
#plt.show()

##======== End of Normal Distribution using Belmont dataset====================


##======== Exponential Distribution uisng Bass Ball dataset ===================

## Draw samples of waiting times: waiting_times
#waiting_times = successive_poisson(764, 715, size=100000)
## Make the histogram
#plt.hist(waiting_times, bins=100, normed=True, histtype='step')
## Label axes
#plt.xlabel('waiting time (number of games)')
#plt.ylabel('PDF')
## Show the plot
#plt.show()

##======== End of Exponential Distribution uisng Bass Ball dataset=============






