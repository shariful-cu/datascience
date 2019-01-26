#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 05:57:30 2019

@author: Shariful
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


sns.set()



#========== Analysis of Iris dataset =============

#---- loading data -----
df_iris = load_iris()
data_iris = df_iris.data
iris_lab = df_iris.target

#---- rough ---
#or df_iris.target_names[1] if x == 1 or
#                             df_iris.target_names[2] if x == 2

#---Histogram----
versicolor_petal_length = data_iris[iris_lab == 1, 2]
n_bins = int(np.sqrt(len(versicolor_petal_length)))
plt.hist(versicolor_petal_length, bins=n_bins)
plt.xlabel('petal length (cm)')
plt.ylabel('count')
plt.show()

##---Swarm plot----
#iris_dataframe = pd.DataFrame(data_iris)
#iris_dataframe.columns = df_iris.feature_names
#iris_lab_cat = list(iris_lab)
#iris_lab_cat = [df_iris.target_names[0] if x == 0 else x for x in iris_lab_cat]
#iris_lab_cat = [df_iris.target_names[1] if x == 1 else x for x in iris_lab_cat]
#iris_lab_cat = [df_iris.target_names[2] if x == 2 else x for x in iris_lab_cat]
#
#iris_dataframe['species'] = iris_lab_cat
#sns.swarmplot(x='species', y='petal length (cm)', data=iris_dataframe)
#plt.xlabel('species')
#plt.ylabel('petal length (cm)')
#plt.show()

#====== End of Analysis of Iris dataset =============


##======== Analysis of  ELection dataset =============
#data_dir = '/Users/Shariful/Documents/GitHubRepo/Datasets'
#df_swing = pd.read_csv(data_dir + '/2008_swing_states.csv')
#
#
##df_swing[['state', 'county', 'dem_share']].head()
#
###---Histogram----
##plt.hist(df_swing['dem_share'])
##plt.xlabel('% of vote for Obama')
##plt.ylabel('number of counties')
##plt.show()
#
##---Swarm plot----
#sns.swarmplot(x='state', y='dem_share', data=df_swing)
#plt.xlabel('state')
#plt.ylabel('% of votes for Obama')
#plt.show()

###====== End of Analysis of  ELection dataset =============




























