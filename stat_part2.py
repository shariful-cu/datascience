#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 05:57:30 2019

@author: Shariful
"""
#---- import necessary modules -----
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.datasets import load_iris
#import datetime
import stat_part1 as utlty_f

#---- global values -----
data_dir = '/Users/Shariful/Documents/GitHubRepo/Datasets'
sns.set()


def bootstrap_replicate_1d(data, func):
    """Generate bootstrap replicate of 1D data."""
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample) 

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""
    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)
    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)
    
    return bs_replicates

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""
    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))
    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)
    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(x))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return bs_slope_reps, bs_intercept_reps

def permutation_sample(data1, data2):
    """Generate a permutation sample from two 1D data sets."""
    # Concatenate the data sets: data
    data = np.concatenate([data1, data2])
    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)
    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[0:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]
    return perm_sample_1, perm_sample_2

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""
    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)
    return diff

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates from two 1d datasets."""
    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)
    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)
        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)
    
    return perm_replicates

#====================== EDA of literacy/fertility data ========================

##--loading data---
#df_literacy = pd.read_csv(data_dir + '/female_literacy_fertility.csv')
#illiteracy = 100 - df_literacy['female literacy']
#fertility = df_literacy['fertility']

#--linear regression using np.polyfit()---
##scatter plot between illiteracy vs fertility 
#plt.plot(illiteracy, fertility, marker='.', linestyle='none')
#plt.margins(0.02)
#plt.xlabel('percent illiterate')
#plt.ylabel('fertility')
##compute and print pearson correlation between illiteracy vs fertility 
#print(utlty_f.pearson_r(illiteracy, fertility))
## Perform a linear regression using np.polyfit(): a, b
#a, b = np.polyfit(illiteracy, fertility, 1)
## Print the results to the screen
#print('slope =', a, 'children per woman / percent illiterate')
#print('intercept =', b, 'children per woman')
## Make theoretical line to plot
#x = np.array([0,100])
#y = a * x + b
## Add regression line to your plot
#plt.plot(x, y)
## Draw the plot
#plt.show()
#
## how did the slope become optimal / residual sum of squares
## Specify slopes to consider: a_vals
#a_vals = np.linspace(0,0.1, 200)
## Initialize sum of square of residuals: rss
#rss = np.empty_like(a_vals)
## Compute sum of square of residuals for each value of a_vals
#for i, a in enumerate(a_vals):
#    rss[i] = np.sum((fertility - a*illiteracy - b)**2)
## Plot the RSS
#plt.plot(a_vals, rss, '-')
#plt.xlabel('slope (children per woman / percent illiterate)')
#plt.ylabel('sum of square of residuals')
#plt.show()

##------pairs bootstrap for linear regression between illiteracy vs fertility --
#bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(illiteracy, \
#                                                        fertility, size=1000)
#print(np.percentile(bs_slope_reps, [2.5, 97.5]))
## Plot the histogram
#_ = plt.hist(bs_slope_reps, bins=50, normed=True)
#_ = plt.xlabel('slope')
#_ = plt.ylabel('PDF')
#plt.show()
#
##Plotting bootstrap regressions
## Generate array of x-values for bootstrap lines: x
#x = np.array([0, 100])
## Plot the bootstrap lines
#for i in range(100):
#    _ = plt.plot(x, 
#                 bs_slope_reps[i] * x + bs_intercept_reps[i],
#                 linewidth=0.5, alpha=0.2, color='red')
## Plot the data
#_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
## Label axes, set the margins, and show the plot
#_ = plt.xlabel('illiteracy')
#_ = plt.ylabel('fertility')
#plt.margins(0.02)
#plt.show()

 
#====== End of Analysis of literacy/fertility data ========================


#================== Linear regression on appropriate Anscombe data ============
"""
- four types of observations with four sets ((x,y),(x,y),(x,y),(x,y)) of data 
  points. 
- The average x and y values for each set is equal
- the regression line is also same for each set 
- the residual sum of squares (rss) is also same for each set
"""
##--loading data---
#df_anscombe = pd.read_csv(data_dir + '/anscombe.csv', header=1, skiprows=0)
#
##---linear regression---
#anscombe_x = []
#anscombe_y = []
#i = 0
#while i < df_anscombe.shape[1]:
#    anscombe_x.append(df_anscombe.iloc[:,i])
#    anscombe_y.append(df_anscombe.iloc[:,i+1])
#    i += 2
#
## Iterate through x,y pairs
#for x, y in zip(anscombe_x, anscombe_y):
#    # Compute the slope and intercept: a, b
#    a, b = np.polyfit(x,y,1)
#    # Print the result
#    print('slope:', a, 'intercept:', b)
#
#x = anscombe_x[0]
#y = anscombe_y[0]
## Perform linear regression: a, b
#a, b = np.polyfit(x,y,1)
## Print the slope and intercept
#print(a, b)
## Generate theoretical x and y data: x_theor, y_theor
#x_theor = np.array([3, 15])
#y_theor = a * x_theor + b
## Plot the Anscombe data and theoretical line
#_ = plt.plot(x, y, marker='.', linestyle='none')
#_ = plt.plot(x_theor, y_theor)
## Label the axes
#plt.xlabel('x')
#plt.ylabel('y')
## Show the plot
#plt.show()

#================== End of Linear regression on appropriate Anscombe data =====


#================== Bootstrap samples on Minchel's speed of light =============

###--loading data---
#df_speed_of_light = pd.read_csv(data_dir + '/michelson_speed_of_light.csv')
##df_speed_of_light.head()
#
#michelson_speed_of_light = df_speed_of_light['velocity of light in air (km/s)']
#bs_sample = np.random.choice(michelson_speed_of_light, size=100)
#
#print(np.mean(bs_sample))
#print(np.std(bs_sample))
#print(np.median(bs_sample))

#================== End of Bootstrap samples on Minchel's speed of light ======


#================== Bootstrap samples on sheffield weather station data =======
"""
- the set of annual rainfall data measured at the Sheffield Weather Station in 
  the UK from 1883 to 2015
- although data is processed but it is not in structured or tabular format
- more preprocessing (transform into tabular format) is needed. 
"""
###--loading data---
#sheffield_weather_station = pd.read_csv(data_dir + \
#                                        '/sheffield_weather_station.csv', \
#                                        skiprows=list(range(0,8)), \
#                                        usecols=[0])
##preprocessed sheffield weather station data
#df_rainfall_np = np.empty((len(sheffield_weather_station), 7))
#for idx, item in sheffield_weather_station.iterrows():
#    item = item[0].replace('---', '-111')
#    values = re.findall(r"[+-]?\d+(?:\.\d+)?", item)
#    df_rainfall_np[idx][:] = np.array(values, dtype=float)
#    
#df_rainfall_np[df_rainfall_np == -111] = np.nan

##--------Bootstrap samples on rainfall (rain) column----
#rainfall = df_rainfall_np[:, 5]
#rainfall = rainfall[~np.isnan(rainfall)]
#rainfall = rainfall[0:134]
#
#for _ in range(50):
#    # Generate bootstrap sample: bs_sample
#    bs_sample = np.random.choice(rainfall, size=rainfall.size)
#    # Compute and plot ECDF from bootstrap sample
#    x, y = utlty_f.ecdf(bs_sample)
#    _ = plt.plot(x, y, marker='.', linestyle='none',
#                 color='gray', alpha=0.1)
#
## Compute and plot ECDF from original data
#x, y = utlty_f.ecdf(rainfall)
#_ = plt.plot(x, y, marker='.')
#
## Make margins and label axes
#plt.margins(0.02)
#_ = plt.xlabel('yearly rainfall (mm)')
#_ = plt.ylabel('ECDF')
## Show the plot
#plt.show()

##-------probabilistic estimate of the mean of annual rainfall using Bootstrap-- 
##-------Replicate--------------------------------------------------------------
#"""
#Verify it with the SEM (Standard Error of the Mean)
#"""
## Take 10,000 bootstrap replicates of the mean: bs_replicates
#bs_replicates = draw_bs_reps(rainfall, np.mean, 10000)
## Compute and print SEM
#sem = np.std(rainfall) / np.sqrt(len(rainfall))
#print(sem)
## Compute and print standard deviation of bootstrap replicates
#bs_std = np.std(bs_replicates)
#print(bs_std)
## Make a histogram of the results
#_ = plt.hist(bs_replicates, bins=50, normed=True)
#_ = plt.xlabel('mean annual rainfall (mm)')
#_ = plt.ylabel('PDF')
## Show the plot
#plt.show()

##-------probabilistic estimate of the variance of annual rainfall using ------- 
##-------Bootstrap Replicate----------------------------------------------------
#"""
#resulting histogram proved that it is not normally distributed, as it has a 
#longer tail to the right.
#"""
## Generate 10,000 bootstrap replicates of the variance: bs_replicates
#bs_replicates = draw_bs_reps(rainfall, np.var, 10000)
## Put the variance in units of square centimeters for convenience
#bs_replicates = bs_replicates / 100
#
## Make a histogram of the results
#_ = plt.hist(bs_replicates, bins=50, normed=True)
#_ = plt.xlabel('variance of annual rainfall (sq. cm)')
#_ = plt.ylabel('PDF')
## Show the plot
#plt.show()

##----------computing the confidence intervals of annual rainfall using ------- 
##-------Bootstrap Replicate----------------------------------------------------
#"""
#one of the many beautiful properties of the bootstrap method is that you can 
#take percentiles of your bootstrap replicates to get your confidence interval
#
#**What is the 95% confidence interval?
#    the 2.5th and 97.5th percentile of your bootstrap replicates 
#"""
#bs_replicates = draw_bs_reps(rainfall, np.mean, 10000)
#conf_intvl = np.percentile(bs_replicates, [2.5, 97.5])

##-----------compute 95% Confidence interval on the rate of no-hitters----------
## Draw samples of waiting times: waiting_times
#waiting_times = utlty_f.successive_poisson(764, 715, size=100000)
## Draw bootstrap replicates of the mean no-hitter time (equal to tau): bs_replicates
#bs_replicates = draw_bs_reps(waiting_times, np.mean, 10000)
## Compute the 95% confidence interval: conf_int
#conf_intvl =  np.percentile(bs_replicates, [2.5, 97.5])
## Print the confidence interval
#print('95% confidence interval =', conf_intvl, 'games')
## Plot the histogram of the replicates
#_ = plt.hist(bs_replicates, bins=50, normed=True)
#_ = plt.xlabel(r'$\tau$ (games)')
#_ = plt.ylabel('PDF')
## Show the plot
#plt.show()

##-----test a hypothesis of two variables are identically distributed using ---
##-----permutation sampling between them ------
#"""
#We will use the Sheffield Weather Station data again, this time considering the
#monthly rainfall in July (a dry month) and November (a wet month). We expect 
#these might be differently distributed, so we will take permutation samples to 
#see how their ECDFs would look if they were identically distributed. 
#
#Discussion:
#    Notice that the permutation samples ECDFs overlap and give a purple haze. 
#    None of the ECDFs from the permutation samples overlap with the observed 
#    data, suggesting that the hypothesis is not commensurate with the data. 
#    July and November rainfall are not identically distributed.
#"""
#rain_june = df_rainfall_np[df_rainfall_np[:,1] == 6, 5]
#rain_june = rain_june[~np.isnan(rain_june)]
#rain_november = df_rainfall_np[df_rainfall_np[:,1] == 11, 5]
#rain_november = rain_november[~np.isnan(rain_november)]
#
#for _ in range(50):
#    # Generate permutation samples
#    perm_sample_1, perm_sample_2 = permutation_sample(rain_june, rain_november)
#    # Compute ECDFs
#    x_1, y_1 = utlty_f.ecdf(perm_sample_1)
#    x_2, y_2 = utlty_f.ecdf(perm_sample_2)
#    # Plot ECDFs of permutation sample
#    _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
#                 color='red', alpha=0.02)
#    _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
#                 color='blue', alpha=0.02)
## Create and plot ECDFs from original data
#x_1, y_1 = utlty_f.ecdf(rain_june)
#x_2, y_2 = utlty_f.ecdf(rain_november)
#_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
#_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')
## Label axes, set margin, and show plot
#plt.margins(0.02)
#_ = plt.xlabel('monthly rainfall (mm)')
#_ = plt.ylabel('ECDF')
#plt.show()

#============ End of Bootstrap samples on sheffield weather station data ======


# =================hypothesis test using frog_tongue dataset ==================
"""
Database Description:
    Kleinteich and Gorb (Sci. Rep., 4, 5225, 2014) performed an interesting 
    experiment with South American horned frogs. They held a plate connected 
    to a force transducer, along with a bait fly, in front of them. They then 
    measured the impact force and adhesive force of the frog's tongue when it 
    struck the target.
    
    Frog A is an adult and Frog B is a juvenile. The researchers measured the 
    impact force of 20 strikes for each frog. In the next exercise, we will 
    test the hypothesis that the two frogs have the same distribution of 
    impact forces. 
Result Analysis:
    The p-value tells you that there is about a 0.6% chance that you would get 
    the difference of means observed in the experiment if frogs were exactly 
    the same. A p-value below 0.01 is typically said to be "statistically 
    significant," but: warning! warning! warning! You have computed a p-value; 
    it is a number. I encourage you not to distill it to a yes-or-no phrase. 
    p = 0.006 and p = 0.000000006 are both said to be "statistically 
    significant," but they are definitely not the same!
"""
##--loading data---
df_frog_tongue = pd.read_csv(data_dir + '/frog_tongue.csv', \
                             skiprows=list(range(14)))
#reading only columns: 'ID' and 'impact force (mN)'
df_frog_tongue = df_frog_tongue.loc[:][['ID', 'impact force (mN)']]
#modify column names
df_frog_tongue.columns = ['ID', 'impact_force']
#convert impact_force from mN to N
df_frog_tongue['impact_force'] = df_frog_tongue['impact_force']/1000
#replace ID values with two groups A(Adult)  & B(Juvenile): 
#'Adult' (ID = I & II) and 'Juvenile' (ID = III & IV)
df_frog_tongue = df_frog_tongue.replace(['I', 'II'], 'A')
df_frog_tongue = df_frog_tongue.replace(['III', 'IV'], 'B')
#read top 20 samples from each group
df_ID_A = df_frog_tongue[df_frog_tongue['ID']=='A']
df_ID_B = df_frog_tongue[df_frog_tongue['ID']=='B']
#merge subset of group A and group B
df_frog_tongue = pd.concat([df_ID_A, df_ID_B])
#
##---EDA before hypothesis testing-----
## Make bee swarm plot
#_ = sns.swarmplot(x='ID', y='impact_force', data=df_frog_tongue)
## Label axes
#_ = plt.xlabel('frog')
#_ = plt.ylabel('impact force (N)')
## Show the plot
#plt.show()
#
##-----Permutation test on frog data----
#force_a = df_ID_A['impact_force']
#force_b = df_ID_B['impact_force']
## Compute difference of mean impact force from experiment: empirical_diff_means
#empirical_diff_means = diff_of_means(force_a, force_b)
## Draw 10,000 permutation replicates: perm_replicates
#perm_replicates = draw_perm_reps(force_a, force_b,
#                                 diff_of_means, size=10000)
## Compute p-value: p
#p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)
## Print the result
#print('p-value =', p)
#
#plt.hist(perm_replicates, bins=50, normed=True)
#_ = plt.xlabel('difference mean impact forces')
#_ = plt.ylabel('PDF')
#
#x = [empirical_diff_means, empirical_diff_means]
#y = [0, 3.2]
#plt.plot(x, y, color='red')
#
#plt.show()

##----A one-sample bootstrap hypothesis test----
#"""
#Another juvenile frog was studied, Frog C, and you want to see if Frog B and 
#Frog C have similar impact forces. Unfortunately, you do not have Frog C's 
#impact forces available, but you know they have a mean of 0.55 N. Because you 
#don't have the original data, you cannot do a permutation test, and you cannot 
#assess the hypothesis that the forces from Frog B and Frog C come from the 
#same distribution.
#
#You will therefore test another, less restrictive hypothesis as:
#
#Test Hypothesis:
#     The mean strike force of Frog B is equal to that of Frog C.
#     
#     To set up the bootstrap hypothesis test, you will take the mean as our 
#     test statistic. Remember, your goal is to calculate the probability of 
#     getting a mean impact force less than or equal to what was observed for 
#     Frog B if the hypothesis that the true mean of Frog B's impact forces is 
#     equal to that of Frog C is true. You first translate all of the data of 
#     Frog B such that the mean is 0.55 N. This involves adding the mean force 
#     of Frog C and subtracting the mean force of Frog B from each measurement 
#     of Frog B. This leaves other properties of Frog B's distribution, such as 
#     the variance, unchanged.
#"""
#force_b = df_ID_B['impact_force']
## another juvenile frog 'C' that has only mean (0.55N) of impact forces but 
## none of his original impact forces data
#force_c_mean = 0.55 
## Make an array of translated impact forces: translated_force_b
#translated_force_b = force_b - np.mean(force_b) + force_c_mean
## Take bootstrap replicates of Frog B's translated impact forces: bs_replicates
#bs_replicates = draw_bs_reps(translated_force_b, np.mean, 10000)
#
##plotting PDF of this permutatoin test
#plt.hist(bs_replicates, bins=50, normed=True)
#_ = plt.xlabel('mean impact forces')
#_ = plt.ylabel('PDF')
#
#x = [np.mean(force_b), np.mean(force_b)]
#y = [0, 12]
#plt.plot(x, y, color='red')
#
#plt.show()
#
## Compute fraction of replicates that are less than the observed Frog B force: p
#p = np.sum(bs_replicates <= np.mean(force_b)) / 10000
#
## Print the p-value
#print('p = ', p)

#-----  A two-sample bootstrap hypothesis test for difference of means ----
"""
Test Hypothesis:
    We now want to test the hypothesis that Frog A and Frog B have the same 
    mean impact force, but not necessarily the same distribution, which is 
    also impossible with a permutation test.
    
Hints:
    To do the two-sample bootstrap test, we shift both arrays to have the same 
    mean, since we are simulating the hypothesis that their means are, in fact, 
    equal. We then draw bootstrap samples out of the shifted arrays and compute 
    the difference in means. This constitutes a bootstrap replicate, and we 
    generate many of them. The p-value is the fraction of replicates with a 
    difference in means greater than or equal to what was observed.
Results Analysis:
    Since the p-value is 0.0, 0% probability that the means force imapcts of 
    Frog A and Frog B are equal or more extreme than the observed difference 
    means of 0.63
    
    Nice work! You got a similar result as when you did the permutation test. 
    Nonetheless, remember that it is important to carefully think about what 
    question you want to ask. Are you only interested in the mean impact force, 
    or in the distribution of impact forces?
"""
# Compute mean of all forces: mean_force
force_a = df_ID_A['impact_force']
force_b = df_ID_B['impact_force']
forces_concat = np.concatenate([force_a, force_b])
mean_force = np.mean(forces_concat)

# Generate shifted arrays
force_a_shifted = force_a - np.mean(force_a) + mean_force
force_b_shifted = force_b - np.mean(force_b) + mean_force 

# Compute 10,000 bootstrap replicates from shifted arrays
bs_replicates_a = draw_bs_reps(force_a_shifted, np.mean, 10000)
bs_replicates_b = draw_bs_reps(force_b_shifted, np.mean, 10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_replicates_a - bs_replicates_b

#plotting PDF of this permutatoin test
plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('difference means of impact forces')
_ = plt.ylabel('PDF')

x = [diff_of_means(force_a, force_b), diff_of_means(force_a, force_b)]
y = [0, 4]
plt.plot(x, y, color='red')
plt.show()

# Compute and print p-value: p
p = np.sum(bs_replicates >= diff_of_means(force_a, force_b)) / 10000
#p = np.sum(bs_replicates <= np.mean(force_b)) / 10000
print('p-value =', p)


# ================ end of hypothesis test using frog_tongue dataset ===========


# =================hypothesis test using swing state election dataset =========

##--loading data---
#df_swing = pd.read_csv(data_dir + '/2008_swing_states.csv')
#
###-----Hypothesis test----
"""
Null Hypothesis:
    the distribution of democretic vote share (% of votes) between these two states 
    are identically distributed
Test Statistic:
    Is the difference means of votes share between these two states 
Resutl Aanlysis:
    Since the p-value is 23%, 23% probability that the difference means of 
    votes shares between these two states are equal or more extreme than the 
    observeed difference means of votes shares between these two states. 
"""
##read share of votes (% of votes) for two states: PA and OH
#dem_share_PA = df_swing[df_swing['state'] == 'PA']['dem_share']
#dem_share_OH = df_swing[df_swing['state'] == 'OH']['dem_share']
## Compute difference of mean votes share from experiment: empirical_diff_means
#empirical_diff_means = diff_of_means(dem_share_PA, dem_share_OH)
## Draw 10,000 permutation replicates: perm_replicates
#perm_replicates = draw_perm_reps(dem_share_PA, dem_share_OH,
#                                 diff_of_means, size=10000)
##plotting PDF of this permutatoin test
#plt.hist(perm_replicates, bins=50, normed=True)
#_ = plt.xlabel('difference mean % of votes shares')
#_ = plt.ylabel('PDF')
#
#x = [empirical_diff_means, empirical_diff_means]
#y = [0, 0.3]
#plt.plot(x, y, color='red')
#
#plt.show()
#
## Compute p-value: p
#p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)
## Print the result
#print('p-value =', p)

# ================ end of hypothesis test using swing state dataset ===========



