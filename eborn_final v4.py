# -*- coding: utf-8 -*-
"""
Eric Born
Class: CS677 - Summer 2
Date: 12 Aug 2019
Final project
Predicting the winning team in the video game League of Legends
"""

import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sys import exit
from sklearn import svm
from sklearn import tree
#from joblib import Parallel, delayed
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LogisticRegression, LinearRegression
#from sklearn.metrics import confusion_matrix, recall_score

# Set display options for dataframes
#pd.set_option('display.max_rows', 100)
#pd.set_option('display.width', 500)
#pd.set_option('display.max_columns', 50)

# set seaborn to dark backgrounds
sns.set_style("darkgrid")

################
# Start data import and cleanup
################
timings_list = []

# start timing
timings_list.append(['global time start', time.time()])
timings_list.append(['clean time start', time.time()])

# setup input directory and filename
data = 'games'
input_dir = r'C:\Users\TomBrody\Desktop\School\677\Final'
csv_file = os.path.join(input_dir, data + '.csv')

# read csv file into dataframe
try:
    lol_df = pd.read_csv(csv_file)
    print('opened file for ticker: ', data, '\n')

except Exception as e:
    print(e)
    exit('Failed to read LoL data from: '+ str(data)+'.csv')

# describe the total rows and columns
print('The total length of the dataframe is', lol_df.shape[0], 'rows',
      'and the width is', lol_df.shape[1], 'columns')

# create a class label using 0 or 1 to indicate winning team
# 0 = team 1 won
# 1 = team 2 won
lol_df['win'] = lol_df['winner'].apply(lambda x: 0 if x == 1 else 1)

# remove columns gameId, creationTime, seasonId and winner
lol_df.drop(lol_df.columns[[0,1,3,4]], axis = 1, inplace = True)

# There are -1's stored in the t1 and t2 ban columns that need to
# be replaced before the chi-squared can be run
# for loop cycles through t1_ban1, t1_ban2, etc. for both teams
# and sets the row to a 0 instead of a -1. goes through team1 and 2
# and all the way up to champ5
try:
    for team in range(1, 3):
        for char in range(1, 6):
            t = 't'+str(team)+'_ban'+str(char)
            print(t, 'had', len(lol_df.loc[lol_df[t] == -1, t]),
                  '-1s replaced with a 0')
            lol_df.loc[lol_df[t] == -1, t] = 0
except Exception as e:
    print(e)
    exit('Failed to replace -1 with 0 in the lol_df')

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

# 58 columns remaining
print(lol_df.head())

# x stores all columns except for the win column
lol_x = lol_df.drop('win', 1)

# y stores only the win column since its used as a predictor
lol_y = lol_df['win']

# setup empty list to store all of the models accuracy
global_accuracy = []

# end timing
timings_list.append(['clean time end', time.time()])

################
# End data import and cleanup
################

################
# Start attribute selection with various methods
################
# start timing
timings_list.append(['features time start', time.time()])

#######
# Start Pearsons corerelation
#######

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

# create dataframe top 5 correlated attributes
pear_five_df = lol_df[['firstInhibitor', 't1_towerKills', 't1_inhibitorKills', 
                      't2_towerKills', 't2_inhibitorKills']]


# create dataframe top 10 correlated attributes
pear_ten_df = lol_df[['firstTower','firstInhibitor', 't1_towerKills',
                       't1_inhibitorKills', 't1_baronKills', 't1_dragonKills',
                       't2_towerKills', 't2_inhibitorKills', 't2_baronKills',
                       't2_dragonKills']]

#######
# End Pearsons corerelation
#######

#######
# Start Ordinary Least Squares
#######

# creates a list of column names
cols = list(lol_x.columns)
# sets a max value
pmax = 1

# while loop that calculates the p values of each attribute using the 
# OLS model and eliminiates the highest value from the list of columns
# loop breaks if all columns remaining have less than 0.05 p value
# or all columns are removed
try:
    while (len(cols)>0):
        p = []
        ols_x1 = lol_x[cols]
        ols_x1 = sm.add_constant(ols_x1)
        model = sm.OLS(lol_y,ols_x1).fit()
        p = pd.Series(model.pvalues.values[1:], index = cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax > 0.05):
            cols.remove(feature_with_p_max)
        else:
            break
except Exception as e:
    print(e)
    exit('Failed to reduce features for ols dataset')
    
# sets and prints the remaining unremoved features
selected_features_BE = cols
print(selected_features_BE)

# creates a dataframe with the ols selected columns
ols_df = lol_df[selected_features_BE]

######
# End Ordinary Least Squares
######

######
# Start Recursive Feature Elimination
######

#####!!!!!!!!!
# only used to determine optimum number of attributes
#####

## Total number of features
#nof_list = np.arange(1,58)            
#high_score = 0
#
## Variable to store the optimum features
#nof = 0           
#score_list = []
#for n in range(len(nof_list)):
#    X_train, X_test, y_train, y_test = train_test_split(lol_x, lol_y, 
#                                            test_size = 0.3, random_state = 0)
#    model = LinearRegression()
#    rfe = RFE(model,nof_list[n])
#    X_train_rfe = rfe.fit_transform(X_train,y_train)
#    X_test_rfe = rfe.transform(X_test)
#    model.fit(X_train_rfe,y_train)
#    score = model.score(X_test_rfe,y_test)
#    score_list.append(score)
#    if(score > high_score):
#        high_score = score
#        nof = nof_list[n]
#
## 39 features score of 0.793475
#print("Optimum number of features: %d" %nof)
#print("Score with %d features: %f" % (nof, high_score))

#####!!!!!!!!!
# only used to determine optimum number of attributes
#####

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

# creates a dataframe with the rfe selected columns
rfe_df = lol_df[selected_features_rfe]

#######
# End Recursive Feature Elimination
#######

#######
# Start lasso method
#######

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

#######
# End lasso method
#######

# end timing
timings_list.append(['features time end', time.time()])

################
# End attribute selection with various methods
################

################
# Start building scaled dataframes
################
# start timing
timings_list.append(['frame build time start', time.time()])

# Setup scalers X datasets
scaler = StandardScaler()
scaler.fit(pear_five_df)
pear_five_df_scaled = scaler.transform(pear_five_df)

# pear_five split dataset into 33% test 66% training
(pear_five_scaled_df_train_x, pear_five_scaled_df_test_x, 
 pear_five_scaled_df_train_y, pear_five_scaled_df_test_y) = (
        train_test_split(pear_five_df_scaled, lol_y, test_size = 0.33, 
                         random_state=1337))

#pear_five_scaled_df_train_x
#pear_five_scaled_df_test_x
#pear_five_scaled_df_train_y
#pear_five_scaled_df_test_y

# Setup scalers X dataset
scaler = StandardScaler()
scaler.fit(pear_ten_df)
pear_ten_df_scaled = scaler.transform(pear_ten_df)

# pear_ten split dataset into 33% test 66% training
(pear_ten_scaled_df_train_x, pear_ten_scaled_df_test_x,
 pear_ten_scaled_df_train_y, pear_ten_scaled_df_test_y) = (
        train_test_split(pear_ten_df_scaled, lol_y, test_size = 0.33, 
                         random_state=1337))

#pear_ten_scaled_df_train_x
#pear_ten_scaled_df_test_x
#pear_ten_scaled_df_train_y
#pear_ten_scaled_df_test_y

# Setup scalers X dataset
scaler = StandardScaler()
scaler.fit(ols_df)
ols_df_scaled = scaler.transform(ols_df)

# ols_df split dataset into 33% test 66% training
(ols_scaled_df_train_x, ols_scaled_df_test_x, ols_scaled_df_train_y,
 ols_scaled_df_test_y) = (
        train_test_split(ols_df_scaled, lol_y, test_size = 0.33, 
                         random_state=1337))

#ols_scaled_df_train_x
#ols_scaled_df_test_x
#ols_scaled_df_train_y
#ols_scaled_df_test_y

# Setup scalers X dataset
scaler = StandardScaler()
scaler.fit(rfe_df)
rfe_df_scaled = scaler.transform(rfe_df)

# ols_df split dataset into 33% test 66% training
(rfe_scaled_df_train_x, rfe_scaled_df_test_x, 
 rfe_scaled_df_train_y, rfe_scaled_df_test_y) = (
        train_test_split(rfe_df_scaled, lol_y, test_size = 0.33, 
                         random_state=1337))
#rfe_scaled_df_train_x
#rfe_scaled_df_test_x
#rfe_scaled_df_train_y
#rfe_scaled_df_test_y

# Setup scalers X dataset
scaler = StandardScaler()
scaler.fit(lasso_df)
lasso_df_scaled = scaler.transform(lasso_df)

# lasso split dataset into 33% test 66% training
(lasso_scaled_df_train_x, lasso_scaled_df_test_x, lasso_scaled_df_train_y, 
lasso_scaled_df_test_y) = (train_test_split(lasso_df_scaled, lol_y, 
                                             test_size = 0.33, 
                                             random_state=1337))
#lasso_scaled_df_train_x
#lasso_scaled_df_test_x
#lasso_scaled_df_train_y
#lasso_scaled_df_test_y

################
# End building scaled dataframes
################

################
# Start building test/train datasets with various attribute selections
################

# dataframes with selected attribtues from 5 methods for attribute eliminiation
#pear_five_df
#pear_ten_df
#ols_df
#rfe_df
#lasso_df

# pear_five split dataset into 33% test 66% training
(pear_five_df_train_x, pear_five_df_test_x, 
pear_five_df_train_y, pear_five_df_test_y) = (
        train_test_split(pear_five_df, lol_y, test_size = 0.33, 
                         random_state=1337))

#pear_five_df_train_x
#pear_five_df_test_x
#pear_five_df_train_y
#pear_five_df_test_y

# pear_ten split dataset into 33% test 66% training
(pear_ten_df_train_x, pear_ten_df_test_x, 
 pear_ten_df_train_y, pear_ten_df_test_y) = (
        train_test_split(pear_ten_df, lol_y, test_size = 0.33, 
                         random_state=1337))

#pear_ten_df_train_x
#pear_ten_df_test_x
#pear_ten_df_train_y
#pear_ten_df_test_y

# ols_df split dataset into 33% test 66% training
ols_df_train_x, ols_df_test_x, ols_df_train_y, ols_df_test_y = (
        train_test_split(ols_df, lol_y, test_size = 0.33, 
                         random_state=1337))

#ols_df_train_x
#ols_df_test_x
#ols_df_train_y
#ols_df_test_y

# ols_df split dataset into 33% test 66% training
rfe_df_train_x, rfe_df_test_x, rfe_df_train_y, rfe_df_test_y = (
        train_test_split(rfe_df, lol_y, test_size = 0.33, 
                         random_state=1337))

#rfe_df_train_x
#rfe_df_test_x
#rfe_df_train_y
#rfe_df_test_y

# ols_df split dataset into 33% test 66% training
lasso_df_train_x, lasso_df_test_x, lasso_df_train_y, lasso_df_test_y = (
        train_test_split(lasso_df, lol_y, test_size = 0.33, 
                         random_state=1337))

#lasso_df_train_x
#lasso_df_test_x
#lasso_df_train_y
#lasso_df_test_y

# ols_df split dataset into 33% test 66% training
full_df_train_x, full_df_test_x, full_df_train_y, full_df_test_y = (
        train_test_split(lol_x, lol_y, test_size = 0.33, 
                         random_state=1337))

#full_df_train_x
#full_df_test_x
#full_df_train_y
#full_df_test_y

# end timing
timings_list.append(['frame build time end', time.time()])

################
# End building test/train datasets with various attribute selections
################

# Create list just for the algorithm durations
algorithm_duration_list = []

################
# Start building non-scaled algorithms
################
# start timing
timings_list.append(['non scaled time start', time.time()])
timings_list.append(['tree time start', time.time()])

#######
# Start decision tree
#######

# start time
algorithm_duration_list.append(time.time()) 

# Create a decisions tree classifier
pear_five_tree_clf = tree.DecisionTreeClassifier(criterion = 'entropy')

# Train the classifier on pearsons top 5 attributes
pear_five_tree_clf = pear_five_tree_clf.fit(pear_five_df_train_x, 
                                            pear_five_df_train_y)

# Predict on pearsons top 5 attributes
pear_five_prediction = pear_five_tree_clf.predict(pear_five_df_test_x)

# end time
algorithm_duration_list.append(time.time()) 

####
# End five
####

####
# Start ten
####

# start time
algorithm_duration_list.append(time.time()) 

# Create a decisions tree classifier
pear_ten_tree_clf = tree.DecisionTreeClassifier(criterion = 'entropy')

# Train the classifier on pearsons top 10 attributes
pear_ten_tree_clf = pear_ten_tree_clf.fit(pear_ten_df_train_x, 
                                            pear_ten_df_train_y)

# Predict on pearsons top 10 attributes
pear_ten_prediction = pear_ten_tree_clf.predict(pear_ten_df_test_x)

# end time
algorithm_duration_list.append(time.time()) 

####
# End ten
####

####
# Start ols_df
####

# start time
algorithm_duration_list.append(time.time()) 

# Create a decisions tree classifier
ols_df_tree_clf = tree.DecisionTreeClassifier(criterion = 'entropy')

# Train the classifier on ols attributes
ols_df_tree_clf = ols_df_tree_clf.fit(ols_df_train_x, 
                                            ols_df_train_y)

# Predict on ols attributes
ols_df_prediction = ols_df_tree_clf.predict(ols_df_test_x)

# end time
algorithm_duration_list.append(time.time()) 


####
# End ols_df
####

####
# Start rfe_df
####

# start time
algorithm_duration_list.append(time.time()) 

# Create a decisions tree classifier
rfe_df_tree_clf = tree.DecisionTreeClassifier(criterion = 'entropy')

# Train the classifier on rfe attributes
rfe_df_tree_clf = rfe_df_tree_clf.fit(rfe_df_train_x, 
                                            rfe_df_train_y)

# Predict on rfe attributes
rfe_df_prediction = rfe_df_tree_clf.predict(rfe_df_test_x)

# end time
algorithm_duration_list.append(time.time()) 

####
# End rfe_df
####

####
# Start lasso_df
####

# start time
algorithm_duration_list.append(time.time()) 

# Create a decisions tree classifier
lasso_df_tree_clf = tree.DecisionTreeClassifier(criterion = 'entropy')

# Train the classifier on lasso attributes
lasso_df_tree_clf = lasso_df_tree_clf.fit(lasso_df_train_x, 
                                            lasso_df_train_y)

# Predict on lasso attributes
lasso_df_prediction = lasso_df_tree_clf.predict(lasso_df_test_x)

# end time
algorithm_duration_list.append(time.time()) 

####
# End lasso_df
####

# Store predictions
global_accuracy.append(100-(round(np.mean(pear_five_prediction 
                                          != pear_five_df_test_y) * 100, 2)))
global_accuracy.append(100-(round(np.mean(pear_ten_prediction 
                                          != pear_ten_df_test_y) * 100, 2)))
global_accuracy.append(100-(round(np.mean(ols_df_prediction 
                                          != lasso_df_test_y) * 100, 2)))
global_accuracy.append(100-(round(np.mean(rfe_df_prediction 
                                          != rfe_df_test_y) * 100, 2)))
global_accuracy.append(100-(round(np.mean(lasso_df_prediction 
                                          != lasso_df_test_y) * 100, 2)))
#######
# End decision tree
#######

# end timing
timings_list.append(['tree time end', time.time()])

#######
# Start naive bayes
#######

# start timing
timings_list.append(['naive time start', time.time()])
algorithm_duration_list.append(time.time()) 

# Create a naive bayes classifier
pear_five_gnb_clf = GaussianNB()

# Train the classifier on pearsons top 5 attributes
pear_five_gnb_clf = pear_five_gnb_clf.fit(pear_five_df_train_x, 
                                          pear_five_df_train_y)

# Predict on pearsons top 5 attributes
pear_five_prediction = pear_five_gnb_clf.predict(pear_five_df_test_x)

# end time
algorithm_duration_list.append(time.time()) 

####
# End five
####

####
# Start ten
####

# start timing
algorithm_duration_list.append(time.time()) 

# Create a naive bayes classifier
pear_ten_gnb_clf = GaussianNB()

# Train the classifier on pearsons top 10 attributes
pear_ten_gnb_clf = pear_ten_gnb_clf.fit(pear_ten_df_train_x, 
                                        pear_ten_df_train_y)

# Predict on pearsons top 10 attributes
pear_ten_prediction = pear_ten_gnb_clf.predict(pear_ten_df_test_x)

# end time
algorithm_duration_list.append(time.time()) 

####
# End ten
####

####
# Start ols_df
####

# start timing
algorithm_duration_list.append(time.time())

# Create a naive bayes classifier
ols_df_gnb_clf = GaussianNB()

# Train the classifier on ols attributes
ols_df_gnb_clf = ols_df_gnb_clf.fit(ols_df_train_x, 
                                    ols_df_train_y)

# Predict on ols attributes
ols_df_prediction = ols_df_gnb_clf.predict(ols_df_test_x)

# end time
algorithm_duration_list.append(time.time()) 

####
# End ols_df
####

####
# Start rfe_df
####

# start timing
algorithm_duration_list.append(time.time()) 

# Create a naive bayes classifier
rfe_df_gnb_clf = GaussianNB()

# Train the classifier on rfe attributes
rfe_df_gnb_clf = rfe_df_gnb_clf.fit(rfe_df_train_x, 
                                    rfe_df_train_y)

# Predict on rfe attributes
rfe_df_prediction = rfe_df_gnb_clf.predict(rfe_df_test_x)

# end time
algorithm_duration_list.append(time.time()) 

####
# End rfe_df
####

####
# Start lasso_df
####

# start timing
algorithm_duration_list.append(time.time()) 

# Create a naive bayes classifier
lasso_df_gnb_clf = GaussianNB()

# Train the classifier on lasso attributes
lasso_df_gnb_clf = lasso_df_gnb_clf.fit(lasso_df_train_x, 
                                        lasso_df_train_y)

# Predict on lasso attributes
lasso_df_prediction = lasso_df_gnb_clf.predict(lasso_df_test_x)

# end time
algorithm_duration_list.append(time.time()) 

####
# End lasso_df
####

# Store predictions
global_accuracy.append(100-(round(np.mean(pear_five_prediction 
                                          != pear_five_df_test_y) * 100, 2)))
global_accuracy.append(100-(round(np.mean(pear_ten_prediction 
                                          != pear_ten_df_test_y) * 100, 2)))
global_accuracy.append(100-(round(np.mean(ols_df_prediction 
                                          != lasso_df_test_y) * 100, 2)))
global_accuracy.append(100-(round(np.mean(rfe_df_prediction 
                                          != rfe_df_test_y) * 100, 2)))
global_accuracy.append(100-(round(np.mean(lasso_df_prediction 
                                          != lasso_df_test_y) * 100, 2)))

#######
# End naive bayes
#######

# end timing
timings_list.append(['naive time end', time.time()])

#######
# Start Random Forest
#######

# start timing
timings_list.append(['forest time start', time.time()])

# Random forest classifiers previously configured using a range from
# 1 to 25 trees and from 1 to 10 depth of each tree. 
# Optimal values for each dataset used below
# set random state to 1337 for repeatability

# Create a list to store the optimal tree and depth values 
# for each random forest classifier

####
# Start pear five dataset
####

# start time
algorithm_duration_list.append(time.time())

#singular RF 
rf_clf = RandomForestClassifier(n_estimators = 8, 
                                    max_depth = 8, criterion ='entropy',
                                    random_state = 1337)
rf_clf.fit(pear_five_df_train_x, pear_five_df_train_y)

# store accuracy
global_accuracy.append(100-(round(np.mean(rf_clf.predict(pear_five_df_test_x) 
                                  != pear_five_df_test_y),2)))

# end time
algorithm_duration_list.append(time.time())
 
####
# End pear five dataset
####

####
# Start pear ten dataset
####

# start time
algorithm_duration_list.append(time.time())

#singular RF 
rf_clf = RandomForestClassifier(n_estimators = 23, 
                                    max_depth = 9, criterion ='entropy',
                                    random_state = 1337)
rf_clf.fit(pear_ten_df_train_x, pear_ten_df_train_y)

# store accuracy
global_accuracy.append(100-(round(np.mean(rf_clf.predict(pear_ten_df_test_x) 
                                  != pear_ten_df_test_y),2)))

# end time
algorithm_duration_list.append(time.time())

####
# End pear ten dataset
####

####
# Start ols dataset
####

# start time
algorithm_duration_list.append(time.time())

#singular RF 
rf_clf = RandomForestClassifier(n_estimators = 25, 
                                    max_depth = 10, criterion ='entropy',
                                    random_state = 1337)
rf_clf.fit(ols_df_train_x, ols_df_train_y)

# store accuracy
global_accuracy.append(100-(round(np.mean(rf_clf.predict(ols_df_test_x) 
                                  != ols_df_test_y),2))) 

# end time
algorithm_duration_list.append(time.time())

####
# End ols dataset
####

####
# Start rfe dataset
####

# start time
algorithm_duration_list.append(time.time())

#singular RF 
rf_clf = RandomForestClassifier(n_estimators = 20, 
                                    max_depth = 9, criterion ='entropy',
                                    random_state = 1337)
rf_clf.fit(rfe_df_train_x, rfe_df_train_y)

# store accuracy
global_accuracy.append(100-(round(np.mean(rf_clf.predict(rfe_df_test_x) 
                                  != rfe_df_test_y),2)))

# end time
algorithm_duration_list.append(time.time())

####
# End rfe dataset
####

####
# Start lasso dataset
####

# start time
algorithm_duration_list.append(time.time())

#singular RF 
rf_clf = RandomForestClassifier(n_estimators = 22, 
                                    max_depth = 10, criterion ='entropy',
                                    random_state = 1337)
rf_clf.fit(lasso_df_train_x, lasso_df_train_y)

# store accuracy
global_accuracy.append(100-(round(np.mean(rf_clf.predict(lasso_df_test_x) 
                                  != lasso_df_test_y),2)))

# end time
algorithm_duration_list.append(time.time())

####
# End lasso dataset
####

####
# Start full dataset
####

# start time
algorithm_duration_list.append(time.time())


#singular RF 
rf_clf = RandomForestClassifier(n_estimators = 25, 
                                    max_depth = 10, criterion ='entropy',
                                    random_state = 1337)
rf_clf.fit(full_df_train_x, full_df_train_y)

# store accuracy
global_accuracy.append(100-(round(np.mean(rf_clf.predict(full_df_test_x) 
                                  != full_df_test_y),2)))

# end time
algorithm_duration_list.append(time.time())

####
# End full dataset
####

#############
# End Random Forest
#############

# end timing
timings_list.append(['forest time end', time.time()])
timings_list.append(['non scaled time end', time.time()])

################
# End building non-scaled algorithms
################

################
# Start building scaled algorithms
################

# start timing
timings_list.append(['scaled time start', time.time()])
timings_list.append(['knn time start', time.time()])

#######
# Start KNN
#######

####
# Start pear-five dataset
####

# initalize knn
knn_classifier = KNeighborsClassifier(n_neighbors = 15)

# Train the classifier
knn_classifier.fit(pear_five_scaled_df_train_x, 
                   pear_five_scaled_df_train_y)

# store accuracy
global_accuracy.append(100-(round(np.mean(knn_classifier.predict(
                                          pear_five_scaled_df_test_x) 
                                          != pear_five_scaled_df_test_y),2)))

# end time
algorithm_duration_list.append(time.time())

####
# end pear-five dataset
####

####
# Start pear-ten dataset
####

# initalize knn
knn_classifier = KNeighborsClassifier(n_neighbors = 9)

# Train the classifier
knn_classifier.fit(pear_ten_scaled_df_train_x, 
                   pear_ten_scaled_df_train_y)
        
# Perform predictions
#pred_k = knn_classifier.predict(pear_ten_scaled_df_test_x)

# store accuracy
global_accuracy.append(100-(round(np.mean(knn_classifier.predict(
                                          pear_ten_scaled_df_test_x) 
                                          != pear_ten_scaled_df_test_y),2)))

####
# end pear-ten dataset
####

####
# Start ols dataset
####

# initalize knn
knn_classifier = KNeighborsClassifier(n_neighbors = 11)

# Train the classifier
knn_classifier.fit(ols_scaled_df_train_x, 
                   ols_scaled_df_train_y)
        
# store accuracy
global_accuracy.append(100-(round(np.mean(knn_classifier.predict(
                                          ols_scaled_df_test_x) 
                                          != ols_scaled_df_test_y),2)))

####
# end ols dataset
####
    
####
# Start rfe dataset
####

# initalize knn
knn_classifier = KNeighborsClassifier(n_neighbors = 15)

# Train the classifier
knn_classifier.fit(rfe_scaled_df_train_x, 
                   rfe_scaled_df_train_y)
        
# store accuracy
global_accuracy.append(100-(round(np.mean(knn_classifier.predict(
                                          rfe_scaled_df_test_x) 
                                          != rfe_scaled_df_test_y),2)))

####
# end rfe dataset
####
    
####
# Start lasso dataset
####

# initalize knn
knn_classifier = KNeighborsClassifier(n_neighbors = 25)

# Train the classifier
knn_classifier.fit(lasso_scaled_df_train_x, 
                   lasso_scaled_df_train_y)
        
# Perform predictions
pred_k = knn_classifier.predict(lasso_scaled_df_test_x)

# Store accuracy
global_accuracy.append(round(sum(pred_k == lasso_scaled_df_test_y) 
                              / len(pred_k) * 100, 2))

####
# end lasso dataset
####

#######
# End KNN
#######

# end timing
timings_list.append(['knn time end', time.time()])

#######
# Start linear SVM
#######

# start timing
timings_list.append(['svm linear time start', time.time()])

####
# Start pear five dataset
####

# create a linear SVM classifier
svm_classifier_linear = svm.SVC(kernel = 'linear')

# fit the classifier on training data
svm_classifier_linear.fit(pear_five_scaled_df_train_x, 
                          pear_five_scaled_df_train_y)

# Predict using 2018 feature data
prediction_linear = svm_classifier_linear.predict(pear_five_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_linear != 
                                   pear_five_scaled_df_test_y) * 100, 2)))

####
# End pear five dataset
####

####
# Start pear ten dataset
####

# create a linear SVM classifier
svm_classifier_linear = svm.SVC(kernel = 'linear')

# fit the classifier on training data
svm_classifier_linear.fit(pear_ten_scaled_df_train_x, 
                          pear_ten_scaled_df_train_y)

# Predict using 2018 feature data
prediction_linear = svm_classifier_linear.predict(pear_ten_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_linear != 
                                   pear_ten_scaled_df_test_y) * 100, 2)))

####
# End pear ten dataset
####

####
# Start ols ten dataset
####

# create a linear SVM classifier
svm_classifier_linear = svm.SVC(kernel = 'linear')

# fit the classifier on training data
svm_classifier_linear.fit(ols_scaled_df_train_x, 
                          ols_scaled_df_train_y)

# Predict using 2018 feature data
prediction_linear = svm_classifier_linear.predict(ols_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_linear != 
                                         ols_scaled_df_test_y) * 100, 2)))

####
# End ols ten dataset
####

####
# Start rfe ten dataset
####

# create a linear SVM classifier
svm_classifier_linear = svm.SVC(kernel = 'linear')

# fit the classifier on training data
svm_classifier_linear.fit(rfe_scaled_df_train_x, 
                          rfe_scaled_df_train_y)

# Predict using 2018 feature data
prediction_linear = svm_classifier_linear.predict(rfe_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_linear != 
                                         rfe_scaled_df_test_y) * 100, 2)))

####
# End rfe ten dataset
####

####
# Start lasso ten dataset
####

# create a linear SVM classifier
svm_classifier_linear = svm.SVC(kernel = 'linear')

# fit the classifier on training data
svm_classifier_linear.fit(lasso_scaled_df_train_x, 
                          lasso_scaled_df_train_y)

# Predict using 2018 feature data
prediction_linear = svm_classifier_linear.predict(lasso_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_linear != 
                                         lasso_scaled_df_test_y) * 100, 2)))

####
# End lasso ten dataset
####

#######
# End linear SVM
#######

# end timing
timings_list.append(['svm linear time end', time.time()])

#######
# Start rbf SVM
#######

# start timing
timings_list.append(['svm rbf svm time start', time.time()])

####
# Start pear five dataset
####

# create a rbf SVM classifier
svm_classifier_rbf = svm.SVC(kernel = 'rbf')

# fit the classifier on training data
svm_classifier_rbf.fit(pear_five_scaled_df_train_x, 
                       pear_five_scaled_df_train_y)

# Predict using 2018 feature data
prediction_rbf = svm_classifier_rbf.predict(pear_five_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_rbf != 
                                   pear_five_scaled_df_test_y) * 100, 2)))

####
# End pear five dataset
####

####
# Start pear ten dataset
####

# create a rbf SVM classifier
svm_classifier_rbf = svm.SVC(kernel = 'rbf')

# fit the classifier on training data
svm_classifier_rbf.fit(pear_ten_scaled_df_train_x, 
                       pear_ten_scaled_df_train_y)

# Predict using 2018 feature data
prediction_rbf = svm_classifier_rbf.predict(pear_ten_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_rbf != 
                                   pear_ten_scaled_df_test_y) * 100, 2)))

####
# End pear ten dataset
####

####
# Start ols ten dataset
####

# create a rbf SVM classifier
svm_classifier_rbf = svm.SVC(kernel = 'rbf')

# fit the classifier on training data
svm_classifier_rbf.fit(ols_scaled_df_train_x, 
                       ols_scaled_df_train_y)

# Predict using 2018 feature data
prediction_rbf = svm_classifier_rbf.predict(ols_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_rbf != 
                                         ols_scaled_df_test_y) * 100, 2)))

####
# End ols ten dataset
####

####
# Start rfe ten dataset
####

# create a rbf SVM classifier
svm_classifier_rbf = svm.SVC(kernel = 'rbf')

# fit the classifier on training data
svm_classifier_rbf.fit(rfe_scaled_df_train_x, 
                       rfe_scaled_df_train_y)

# Predict using 2018 feature data
prediction_rbf = svm_classifier_rbf.predict(rfe_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_rbf != 
                                         rfe_scaled_df_test_y) * 100, 2)))

####
# End rfe ten dataset
####

####
# Start lasso ten dataset
####

# create a rbf SVM classifier
svm_classifier_rbf = svm.SVC(kernel = 'rbf')

# fit the classifier on training data
svm_classifier_rbf.fit(lasso_scaled_df_train_x, 
                       lasso_scaled_df_train_y)

# Predict using 2018 feature data
prediction_rbf = svm_classifier_rbf.predict(lasso_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_rbf != 
                                         lasso_scaled_df_test_y) * 100, 2)))

####
# End lasso ten dataset
####

#######
# End rbf SVM
#######

# end timing
timings_list.append(['svm rbf svm time end', time.time()])

#######
# Start poly SVM
#######

# start timing
timings_list.append(['svm poly time start', time.time()])

####
# Start pear five dataset
####

# create a poly SVM classifier
svm_classifier_poly = svm.SVC(kernel = 'poly')

# fit the classifier on training data
svm_classifier_poly.fit(pear_five_scaled_df_train_x, 
                        pear_five_scaled_df_train_y)

# Predict using 2018 feature data
prediction_poly = svm_classifier_poly.predict(pear_five_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_poly != 
                                   pear_five_scaled_df_test_y) * 100, 2)))

####
# End pear five dataset
####

####
# Start pear ten dataset
####

# create a poly SVM classifier
svm_classifier_poly = svm.SVC(kernel = 'poly')

# fit the classifier on training data
svm_classifier_poly.fit(pear_ten_scaled_df_train_x, 
                        pear_ten_scaled_df_train_y)

# Predict using 2018 feature data
prediction_poly = svm_classifier_poly.predict(pear_ten_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_poly != 
                                   pear_ten_scaled_df_test_y) * 100, 2)))

####
# End pear ten dataset
####

####
# Start ols dataset
####

# create a poly SVM classifier
svm_classifier_poly = svm.SVC(kernel = 'poly')

# fit the classifier on training data
svm_classifier_poly.fit(ols_scaled_df_train_x, 
                        ols_scaled_df_train_y)

# Predict using 2018 feature data
prediction_poly = svm_classifier_poly.predict(ols_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_poly != 
                                         ols_scaled_df_test_y) * 100, 2)))

####
# End ols dataset
####

####
# Start rfe dataset
####

# create a poly SVM classifier
svm_classifier_poly = svm.SVC(kernel = 'poly')

# fit the classifier on training data
svm_classifier_poly.fit(rfe_scaled_df_train_x, 
                        rfe_scaled_df_train_y)

# Predict using 2018 feature data
prediction_poly = svm_classifier_poly.predict(rfe_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_poly != 
                                         rfe_scaled_df_test_y) * 100, 2)))

####
# End rfe dataset
####

####
# Start lasso dataset
####

# create a poly SVM classifier
svm_classifier_poly = svm.SVC(kernel = 'poly')

# fit the classifier on training data
svm_classifier_poly.fit(lasso_scaled_df_train_x, 
                        lasso_scaled_df_train_y)

# Predict using 2018 feature data
prediction_poly = svm_classifier_poly.predict(lasso_scaled_df_test_x)

# calculate error rate
global_accuracy.append(100-(round(np.mean(prediction_poly != 
                                         lasso_scaled_df_test_y) * 100, 2)))

####
# End lasso dataset
####

#######
# End poly SVM
#######

#############
# End SVM
#############

# end timing
timings_list.append(['svm poly time end', time.time()])

#############
# Start log regression liblinear solver
#############

# start timing
timings_list.append(['log lib time start', time.time()])

####
# Start pear five dataset
####

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'liblinear')

# Train the classifier on 2017 data
log_reg_classifier.fit(pear_five_scaled_df_train_x, 
                       pear_five_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(pear_five_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              pear_five_scaled_df_test_y) * 100, 2))

####
# End pear five dataset
####

####
# Start pear ten dataset
####

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'liblinear')

# Train the classifier on 2017 data
log_reg_classifier.fit(pear_ten_scaled_df_train_x, 
                       pear_ten_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(pear_ten_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              pear_ten_scaled_df_test_y) * 100, 2))

####
# End pear ten dataset
####

####
# Start ols dataset
####

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'liblinear')

# Train the classifier on 2017 data
log_reg_classifier.fit(ols_scaled_df_train_x, 
                       ols_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(ols_scaled_df_test_x)


# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              ols_scaled_df_test_y) * 100, 2))

####
# End ols dataset
####

####
# Start rfe dataset
####

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'liblinear')

# Train the classifier on 2017 data
log_reg_classifier.fit(rfe_scaled_df_train_x, 
                       rfe_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(rfe_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              rfe_scaled_df_test_y) * 100, 2))

####
# End rfe dataset
####

####
# Start lasso dataset
####

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'liblinear')

# Train the classifier on 2017 data
log_reg_classifier.fit(lasso_scaled_df_train_x, 
                       lasso_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(lasso_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              lasso_scaled_df_test_y) * 100, 2))

####
# End lasso dataset
####

#############
# End log regression liblinear solver
#############

# end timing
timings_list.append(['log lib time end', time.time()])

#############
# Start log regression sag solver
#############

# start timing
timings_list.append(['log sag time start', time.time()])

####
# Start pear five dataset
####

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'sag')

# Train the classifier on 2017 data
log_reg_classifier.fit(pear_five_scaled_df_train_x, 
                       pear_five_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(pear_five_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              pear_five_scaled_df_test_y) * 100, 2))

####
# End pear five dataset
####

####
# Start pear ten dataset
####

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'sag')

# Train the classifier on 2017 data
log_reg_classifier.fit(pear_ten_scaled_df_train_x, 
                       pear_ten_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(pear_ten_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              pear_ten_scaled_df_test_y) * 100, 2))

####
# End pear ten dataset
####

####
# Start ols dataset
####

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'sag')

# Train the classifier on 2017 data
log_reg_classifier.fit(ols_scaled_df_train_x, 
                       ols_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(ols_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              ols_scaled_df_test_y) * 100, 2))

####
# End ols dataset
####

####
# Start rfe dataset
####

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'sag')

# Train the classifier on 2017 data
log_reg_classifier.fit(rfe_scaled_df_train_x, 
                       rfe_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(rfe_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              rfe_scaled_df_test_y) * 100, 2))

####
# End rfe dataset
####

####
# Start lasso dataset
####

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'sag')

# Train the classifier on 2017 data
log_reg_classifier.fit(lasso_scaled_df_train_x, 
                       lasso_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(lasso_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              lasso_scaled_df_test_y) * 100, 2))

####
# End lasso dataset
####

#############
# End log regression sag solver
#############

# end timing
timings_list.append(['log sag time end', time.time()])

#############
# Start log regression newton-cg solver
#############

# start timing
timings_list.append(['log newt time start', time.time()])

####
# Start pear five dataset
####

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'newton-cg')

# Train the classifier on 2017 data
log_reg_classifier.fit(pear_five_scaled_df_train_x, 
                       pear_five_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(pear_five_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              pear_five_scaled_df_test_y) * 100, 2))

####
# End pear five dataset
####

####
# Start pear ten dataset
####

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'newton-cg')

# Train the classifier on 2017 data
log_reg_classifier.fit(pear_ten_scaled_df_train_x, 
                       pear_ten_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(pear_ten_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              pear_ten_scaled_df_test_y) * 100, 2))

####
# End pear ten dataset
####

####
# Start ols dataset
####

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'newton-cg')

# Train the classifier on 2017 data
log_reg_classifier.fit(ols_scaled_df_train_x, 
                       ols_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(ols_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              ols_scaled_df_test_y) * 100, 2))

####
# End ols dataset
####

####
# Start rfe dataset
####

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'newton-cg')

# Train the classifier on 2017 data
log_reg_classifier.fit(rfe_scaled_df_train_x, 
                       rfe_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(rfe_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              rfe_scaled_df_test_y) * 100, 2))

####
# End rfe dataset
####

####
# Start lasso dataset
####

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'newton-cg')

# Train the classifier on 2017 data
log_reg_classifier.fit(lasso_scaled_df_train_x, 
                       lasso_scaled_df_train_y)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(lasso_scaled_df_test_x)

# append the models accruacy to the accuracy list
global_accuracy.append(round(np.mean(prediction == 
                              lasso_scaled_df_test_y) * 100, 2))

####
# End lasso dataset
####

#############
# End log regression newton-cg solver
#############

# end timing
timings_list.append(['log newt time end', time.time()])

################
# End building scaled algorithms
################
timings_list.append(['scaled time end', time.time()])

########
# Start counts of the win totals for team 1 and team 2
########

# store wins in variable for team 1
team1 = sum(lol_df.win == 0)
pear_five_train_team1 = sum(pear_five_df_train_y == 0)
pear_five_test_team1  = sum(pear_five_df_test_y  == 0)
pear_ten_train_team1  = sum(pear_ten_df_train_y  == 0)
pear_ten_test_team1   = sum(pear_ten_df_test_y   == 0)
ols_train_team1       = sum(ols_df_train_y       == 0)
ols_test_team1        = sum(ols_df_test_y        == 0)
rfe_train_team1       = sum(rfe_df_train_y       == 0)
rfe_test_team1        = sum(rfe_df_test_y        == 0)
lasso_train_team1     = sum(lasso_df_train_y     == 0)
lasso_test_team1      = sum(lasso_df_test_y      == 0)

# store wins in variable for team 2
team2 = sum(lol_df.win == 1)
pear_five_train_team2 = sum(pear_five_df_train_y == 1)
pear_five_test_team2  = sum(pear_five_df_test_y  == 1)
pear_ten_train_team2  = sum(pear_ten_df_train_y  == 1)
pear_ten_test_team2   = sum(pear_ten_df_test_y   == 1)
ols_train_team2       = sum(ols_df_train_y       == 1)
ols_test_team2        = sum(ols_df_test_y        == 1)
rfe_train_team2       = sum(rfe_df_train_y       == 1)
rfe_test_team2        = sum(rfe_df_test_y        == 1)
lasso_train_team2     = sum(lasso_df_train_y     == 1)
lasso_test_team2      = sum(lasso_df_test_y      == 1)

# create a ratio of the wins
ratio = round(team1 / team2, 4)

pear_five_train_ratio = round(pear_five_train_team1 / pear_five_train_team2, 4)
pear_five_test_ratio  = round(pear_five_test_team1 / pear_five_test_team2, 4)

pear_ten_train_ratio = round(pear_five_train_team1 / pear_five_train_team2, 4)
pear_ten_test_ratio  = round(pear_five_test_team1 / pear_five_test_team2, 4)

ols_train_ratio = round(ols_train_team1 / ols_train_team2, 4)
ols_test_ratio  = round(ols_test_team1 / ols_test_team2, 4)

rfe_train_ratio = round(rfe_train_team1 / rfe_train_team2, 4)
rfe_test_ratio  = round(rfe_test_team1 / rfe_test_team2, 4)

lasso_train_ratio = round(lasso_train_team1 / lasso_train_team2, 4)
lasso_test_ratio  = round(lasso_test_team1 / lasso_test_team2, 4)

# Print win ratios
print('\nOriginal dataset win ratios\n','team 1 : team 2\n', str(ratio)+
      ' :   1')
print('\nPearson five training win ratios\n','team 1 : team 2\n', 
      str(pear_five_train_ratio)+' :   1')
print('\nPearson five test win ratios\n','team 1 : team 2\n', 
      str(pear_five_test_ratio)+' :   1')
print('\nPearson ten training win ratios\n','team 1 : team 2\n', 
      str(pear_ten_train_ratio)+' :   1')
print('\nPearson ten test win ratios\n','team 1 : team 2\n', 
      str(pear_ten_test_ratio)+' :   1')
print('\nOls training win ratios\n','team 1 : team 2\n', 
      str(ols_train_ratio)+' :   1')
print('\nOls test win ratios\n','team 1 : team 2\n', 
      str(ols_test_ratio)+' :   1')
print('\nRfe training win ratios\n','team 1 : team 2\n', 
      str(rfe_train_ratio)+' :   1')
print('\nRfe test win ratios\n','team 1 : team 2\n', 
      str(rfe_test_ratio)+' :   1')
print('\nLasso training win ratios\n','team 1 : team 2\n', 
      str(lasso_train_ratio)+' :   1')
print('\nLasso test win ratios\n','team 1 : team 2\n', 
      str(lasso_test_ratio)+' :   1')

########
# End counts of the win totals for team 1 and team 2
########

####
# Start prediction prints
####

# create a list containing the attribute reduction methods
attributes = ['Pearson Five', 'Pearson Ten', 'OLS', 'RFE', 'Lasso']

# create a list containing the classifier names
classifiers = ['Decision Tree', 'Naive Bayes', 'Random Forest', 'KNN', 'SVM', 
               'SVM', 'SVM', 'Logistic Regression', 'Logistic Regression',
               'Logistic Regression']

# Creates a dataframe containing information about the classifiers and accuracy
prediction_df = pd.DataFrame(columns =['classifier', 'details', 'attributes',
                                       'accuracy'])

# Build out a dataframe to store the classifiers and their accuracy
for i in range(0, len(classifiers)):
    for k in range(0, len(attributes)):
        prediction_df = prediction_df.append({'classifier' : classifiers[i],
                                      'details' : 'None',
                                      'attributes' : attributes[k],
                                      'accuracy' : 0}, 
                                      ignore_index=True)

# Move indexes down 1 starting at 15 to add Random forest with full dataset
prediction_df.index = (prediction_df.index[:15].tolist() + 
                      (prediction_df.index[15:] + 1).tolist())

# Adds Random forest with full dataset to the dataframe
prediction_df.loc[15] = ['Random Forest', 'None', 'Full', 0]

# reorders the indexes after the insert
prediction_df = prediction_df.sort_index()

# Updates the accuracy
prediction_df['accuracy'] = global_accuracy

#!!!! ADD DETAIL OF SOLVER AND KERNEL TO prediction_df !!!!




# Finds the most accuracy algorithm
final_classifier = prediction_df['classifier'][prediction_df['accuracy'] == 
              max(prediction_df.accuracy)].values[0]

final_details = prediction_df['details'][prediction_df['accuracy'] == 
              max(prediction_df.accuracy)].values[0]

final_attributes = prediction_df['attributes'][prediction_df['accuracy'] == 
              max(prediction_df.accuracy)].values[0]

final_accuracy = prediction_df['accuracy'][prediction_df['accuracy'] == 
              max(prediction_df.accuracy)].values[0]

# print max accuracy
if final_details == 'None':
    print('The best classifier was the', final_classifier, 'using the',
          final_attributes, 'attribute set and an accuracy of', final_accuracy,
          '%')
else:
    print('The best classifier was the', final_classifier, 'with', 
          final_details, 'using the', final_attributes,
           'attribute set and an accuracy of', final_accuracy,
          '%')

timings_list.append(['global time end', time.time()]) 
####
# End prediction prints
####