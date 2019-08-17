# -*- coding: utf-8 -*-
"""
Eric Born
Class: CS677 - Summer 2
Date: 12 Aug 2019
Final project
Predicting the winning team in the video game League of Legends
"""

import os
from sys import exit
from matplotlib import rcParams, pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import confusion_matrix, recall_score
#from sklearn.model_selection import StratifiedKFold

sns.set_style("darkgrid")

# setup input directory and filename
data = 'games'
input_dir = r'C:\Users\TomBrody\Desktop\School\677\Final'
ticker_file = os.path.join(input_dir, data + '.csv')

# read csv file into dataframe
try:
    lol_df = pd.read_csv(ticker_file)
    print('opened file for ticker: ', data,'\n')

except Exception as e:
    print(e)
    exit('failed to read LoL data from: '+ str(data)+'.csv')

# describe the total rows and columns
print('The total length of the dataframe is', lol_df.shape[0], 'rows',
      'and the width is', lol_df.shape[1], 'columns')

# create a class label using 0 or 1 to indicate winning team
# 0 = team 1 won
# 1 = team 2 won
lol_df['win'] = lol_df['winner'].apply(lambda x: 0 if x == 1 else 1)

# remove columns gameId, creationTime, seasonId and winner
lol_df.drop(lol_df.columns[[0,1,3,4]], 
                     axis = 1, inplace = True)

# there are -1's stored in the t1 and t2 ban columns that need to
# be replaced before the chi-squared can be run
# for loop cycles through t1_ban1, t1_ban2, etc. for both teams
# and sets the row to a 0 instead of a -1. goes through team1 and 2
# and all the way up to champ5
for team in range(1, 3):
    for char in range(1, 6):
        t = 't'+str(team)+'_ban'+str(char)
        print(t, 'had', len(lol_df.loc[lol_df[t] == -1, t]),
              '-1s replaced with a 0')
        lol_df.loc[lol_df[t] == -1, t] = 0

## write modified data to csv
## desired csv filename
#name = 'LoL'
#
## save directory
#input_dir = r'C:\Users\TomBrody\Desktop\School\677\Final'
#
## Create an output file name
#output_file = os.path.join(input_dir, name + '.csv')
#
## write df to csv
#lol_df.to_csv(output_file, index=False)

# view row 89, all columns
#lol_df.iloc[89,:]

# store wins in variable for each time
team1 = sum(lol_df.win == 0)
team2 = sum(lol_df.win == 1)

# create a ratio of the wins
ratio = round(team1 / team2, 4)
print('\nTeam win ratios\n','team 1 : team 2\n', str(ratio)+' :   1')

# 58 columns remaining
lol_df.head()

# x stores all columns except for the win column
lol_x = lol_df.drop('win',1)

# y stores only the win column since its used as a predictor
lol_y = lol_df['win']

########
# Start Pearsons corerelation
########

# create a correlation object
cor = lol_df.corr()

# correlation with output variable
cor_target = abs(cor['win'])

# selecting features correlated greater than 0.5
relevant_features_five = cor_target[cor_target>0.5]

# second set of features correlated greater than 0.35
relevant_features_ten = cor_target[cor_target > 0.35]

# results for the top 5 and top 10 attributes
print(relevant_features_five)
print(relevant_features_ten)


#!!!! Choosing not to eliminate these attributes since we would only
# be left with 2 attributes for the pearson 5!!!!!

# reviewing the correlation between these elements to
# determine if any should be eliminated. Attributes close to 0 have
# little correlation. Attributes closer to -1 or 1 have high correlation
# low
#print(lol_df[["t1_towerKills","firstInhibitor"]].corr())
#
## mid
#print(lol_df[["t1_towerKills","t2_towerKills"]].corr())
#
## high correlation
## remove t1_inhibitorKills, t2_inhibitorKills, firstInhibitor
#print(lol_df[["t1_towerKills","t1_inhibitorKills"]].corr())
#print(lol_df[["t2_towerKills","t2_inhibitorKills"]].corr())
#print(lol_df[["t2_towerKills","firstInhibitor"]].corr())

# create dataframe top 5 correlated attributes
pear_five_df = lol_df[['firstInhibitor', 't1_towerKills', 't1_inhibitorKills', 
                      't2_towerKills', 't2_inhibitorKills', 'win']]

# create dataframe top 10 correlated attributes
pear_ten_df = lol_df[['firstTower','firstInhibitor', 't1_towerKills',
                       't1_inhibitorKills', 't1_baronKills', 't1_dragonKills',
                       't2_towerKills', 't2_inhibitorKills', 't2_baronKills',
                       't2_dragonKills', 'win']]

########
# End Pearsons corerelation
########

########
# Start Ordinary Least Squares
########

# creates a list of column names
cols = list(lol_x.columns)
# sets a max value
pmax = 1

# while loop that calculates the p values of each attribute using the 
# OLS model and eliminiates the highest value from the list of columns
# loop breaks if all columns remaining have less than 0.05 p value
# or all columns are removed
while (len(cols)>0):
    p = []
    ols_x1 = lol_x[cols]
    ols_x1 = sm.add_constant(ols_x1)
    model = sm.OLS(lol_y,ols_x1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break

# sets and prints the remaining unremoved features
selected_features_BE = cols
print(selected_features_BE)

# Appends win column for prediction purposes
selected_features_BE.append('win')

# creates a dataframe with the ols selected columns
ols_df = lol_df[selected_features_BE]

########
# End Ordinary Least Squares
########

########
# Start Recursive Feature Elimination
########

# Total number of features
nof_list = np.arange(1,58)            
high_score = 0

# Variable to store the optimum features
nof = 0           
score_list = []
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(lol_x, lol_y, 
                                            test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score > high_score):
        high_score = score
        nof = nof_list[n]

# 39 features score of 0.793475
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

# setup column list and regression model
cols = list(lol_x.columns)
model = LinearRegression()

#Initializing RFE model with 39 features
rfe = RFE(model, 39)   
          
#Transforming data using RFE
X_rfe = rfe.fit_transform(lol_x,lol_y)  

#Fitting the data to model
model.fit(X_rfe,lol_y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index

# output the selected features
print(selected_features_rfe)

# create index object for win column
win_df = pd.Index(['win'])

# Appends win column for prediction purposes
selected_features_rfe = selected_features_rfe.append([win_df])

# creates a dataframe with the ols selected columns
rfe_df = lol_df[selected_features_rfe]

########
# End Recursive Feature Elimination
########

########
# Start lasso method
########

# build the model with a cross validation set to 5
reg = LassoCV(cv = 5)

# fit the model
reg.fit(lol_x, lol_y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(lol_x, lol_y))

# build the coefficients between each attribute
coef = pd.Series(reg.coef_, index = lol_x.columns)

# output total attributes chosen and discarded
print("Lasso picked " + str(sum(coef != 0)) +
      " attributes and eliminated the other " +
      str(sum(coef == 0)) + " variables")


# creates a dataframe based on the 32 columns selected from lasso
lasso_df = lol_df[coef[coef.values != 0].index]

lasso_df['win'] = lol_df.win

########
# End lasso method
########

pear_five_df
pear_ten_df
ols_df
rfe_df
lasso_df



#############
# Chi-squared
# store dataframe values into an array for faster computation
#lol_array = lol_df.values
#
## x stores all columns except for the win column
#chi_x = lol_array[:,0:57]
#
## y stores only the win column since its used as a predictor
#chi_y = lol_array[:,57]
#
## implement chi-squared to evaluate feature significance
## k = 15 is chosing the top 15 attributes
#chi_test = SelectKBest(score_func = chi2, k = 15)
#chi_fit = chi_test.fit(chi_x, chi_y)
#
## set the precision level
#np.set_printoptions(precision = 3)
#
## creates a new array containing the feature set reduced to the best 15
#features = chi_fit.transform(chi_x)
#
#
#print(features[0:16,:])


# split dataset into 33% test 66% training
#lol_df_train, lol_df_test = train_test_split