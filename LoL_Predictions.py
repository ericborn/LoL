# -*- coding: utf-8 -*-
"""
Eric Born
Date: 12 Aug 2019
Predicting the winning team in the video game League of Legends
utilizing various machine learning algorithms
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
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, recall_score,\
                            classification_report

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
timings_list.append(['Global duration:', time.time()])
timings_list.append(['Data clean up duration:', time.time()])

# setup input directory and filename
data = 'LoL'
input_dir = r'C:\Code projects\BU\677 - Data Science with py\Final'
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
timings_list.append(['Features selection duration:', time.time()])

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

# create dataframe top 5 correlated attributes
pear_five_df = lol_df[['firstInhibitor', 't1_towerKills', 't1_inhibitorKills', 
                      't2_towerKills', 't2_inhibitorKills']]


# create dataframe top 10 correlated attributes
pear_ten_df = lol_df[['firstTower','firstInhibitor', 't1_towerKills',
                       't1_inhibitorKills', 't1_baronKills', 't1_dragonKills',
                       't2_towerKills', 't2_inhibitorKills', 't2_baronKills',
                       't2_dragonKills']]

# results for the top 5 and top 10 attributes
print('Pearsons top 5 attributes:')
print(list(pear_five_df.columns), '\n')

print('Pearsons top 10 attributes:')
print(list(pear_ten_df.columns), '\n')

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
        ols_x1 = sm.add_constant(lol_x[cols].values)
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

print('Total features:', len(selected_features_rfe),
    '\nOrdinary Least Squares features:\n', selected_features_rfe)

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

# build the coefficients between each attribute
coef = pd.Series(reg.coef_, index = lol_x.columns)

# creates a dataframe based on the 32 columns selected from lasso
lasso_df = lol_df[coef[coef.values != 0].index]

# output total attributes chosen and discarded
print("Total Features: " + str(sum(coef != 0)),
      '\nLasso features:\n',list(lasso_df.columns))

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
timings_list.append(['Dataframe build duration:', time.time()])

# Setup scalers X datasets
scaler = StandardScaler()
scaler.fit(pear_five_df)
pear_five_df_scaled = scaler.transform(pear_five_df)

# pear_five split dataset into 33% test 66% training
(pear_five_scaled_df_train_x, pear_five_scaled_df_test_x, 
 pear_five_scaled_df_train_y, pear_five_scaled_df_test_y) = (
        train_test_split(pear_five_df_scaled, lol_y, test_size = 0.333, 
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
        train_test_split(pear_ten_df_scaled, lol_y, test_size = 0.333, 
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
        train_test_split(ols_df_scaled, lol_y, test_size = 0.333, 
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
        train_test_split(rfe_df_scaled, lol_y, test_size = 0.333, 
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
                                             test_size = 0.333, 
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
        train_test_split(pear_five_df, lol_y, test_size = 0.333, 
                         random_state=1337))

#pear_five_df_train_x
#pear_five_df_test_x
#pear_five_df_train_y
#pear_five_df_test_y

# pear_ten split dataset into 33% test 66% training
(pear_ten_df_train_x, pear_ten_df_test_x, 
 pear_ten_df_train_y, pear_ten_df_test_y) = (
        train_test_split(pear_ten_df, lol_y, test_size = 0.333, 
                         random_state=1337))

#pear_ten_df_train_x
#pear_ten_df_test_x
#pear_ten_df_train_y
#pear_ten_df_test_y

# ols_df split dataset into 33% test 66% training
ols_df_train_x, ols_df_test_x, ols_df_train_y, ols_df_test_y = (
        train_test_split(ols_df, lol_y, test_size = 0.333, 
                         random_state=1337))

#ols_df_train_x
#ols_df_test_x
#ols_df_train_y
#ols_df_test_y

# ols_df split dataset into 33% test 66% training
rfe_df_train_x, rfe_df_test_x, rfe_df_train_y, rfe_df_test_y = (
        train_test_split(rfe_df, lol_y, test_size = 0.333, 
                         random_state=1337))

#rfe_df_train_x
#rfe_df_test_x
#rfe_df_train_y
#rfe_df_test_y

# ols_df split dataset into 33% test 66% training
lasso_df_train_x, lasso_df_test_x, lasso_df_train_y, lasso_df_test_y = (
        train_test_split(lasso_df, lol_y, test_size = 0.333, 
                         random_state=1337))

#lasso_df_train_x
#lasso_df_test_x
#lasso_df_train_y
#lasso_df_test_y

# ols_df split dataset into 33% test 66% training
full_df_train_x, full_df_test_x, full_df_train_y, full_df_test_y = (
        train_test_split(lol_x, lol_y, test_size = 0.333, 
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
timings_list.append(['Non-scaled duration:', time.time()])
timings_list.append(['Decision tree duration:', time.time()])

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
                                          != ols_df_test_y) * 100, 2)))
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
timings_list.append(['Naive Bayes duration:', time.time()])
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
                                          != ols_df_test_y) * 100, 2)))
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

####
# Start RF accuracy tests
####

# Random forest classifiers using a range from
# 1 to 25 trees and from 1 to 10 depth of each tree
# Set random state to 1337 for repeatability

# Create a list to store the optimal tree and depth values 
# for each random forest classifier
trees_depth = []

# create an empty list to store the rf accuracy at various settings
rf_accuracy = []

# setup an empty dataframe for the rf tests
rf_accuracy_df = pd.DataFrame()

####
# Start pear five dataSet
####

pred_list = []

# RF with iterator
for trees in range(1, 26):
    for depth in range(1, 11):
        rf_clf_test = RandomForestClassifier(n_estimators = trees, 
                                    max_depth = depth, criterion ='entropy',
                                    random_state = 1337)
        rf_clf_test.fit(pear_five_df_train_x, pear_five_df_train_y)
        pred_list.append([trees, depth, 
                    round(np.mean(rf_clf_test.predict(pear_five_df_test_x) 
                    == pear_five_df_test_y) 
                    * 100, 2), 'pear_five'])
         
#create a dataframe from the classifer data
forest_df_1 = pd.DataFrame(pred_list, columns = ['Estimators', 'Depth',
                                                 'Accuracy', 'Set'])

# append forest 1 to full df
rf_accuracy_df = rf_accuracy_df.append(forest_df_1)

# store the lowest error rate value from the classifier
ind = forest_df_1.loc[forest_df_1['Accuracy'] == 
                      max(forest_df_1.Accuracy)].values

# pull out the number of trees and depth
trees_depth.append(int(ind.item(0)))
trees_depth.append(int(ind.item(1)))

# append the models accruacy to the accuracy list
rf_accuracy.append(round(ind.item(2), 2))

print('Pearsons Five:\nOptimal trees:', trees_depth[0],
      '\nOptimal depth:', trees_depth[1])

####
# End pear five dataSet
####

####
# Start pear ten dataSet
####

pred_list = []

for trees in range(1, 26):
    for depth in range(1, 11):
        rf_clf_test = RandomForestClassifier(n_estimators = trees, 
                                    max_depth = depth, criterion ='entropy',
                                    random_state = 1337)
        rf_clf_test.fit(pear_ten_df_train_x, pear_ten_df_train_y)
        pred_list.append([trees, depth, 
                    round(np.mean(rf_clf_test.predict(pear_ten_df_test_x) 
                    == pear_ten_df_test_y) 
                    * 100, 2), 'pear_ten'])

# create a dataframe from the classifer data
forest_df_2 = pd.DataFrame(pred_list, columns = ['Estimators', 'Depth',
                                                 'Accuracy', 'Set'])

# append forest 2 to full df
rf_accuracy_df = rf_accuracy_df.append(forest_df_2)

# store the lowest error rate value from the classifier
ind = forest_df_2.loc[forest_df_2['Accuracy'] == 
                      max(forest_df_2.Accuracy)].values

# pull out the number of trees and depth
trees_depth.append(int(ind.item(0)))
trees_depth.append(int(ind.item(1)))

# append the models accruacy to the accuracy list
rf_accuracy.append(round(ind.item(2), 2))

print('Pearsons Ten:\nOptimal trees:', trees_depth[2],
      '\nOptimal depth:', trees_depth[3])

####
# End pear ten dataSet
####

####
# Start ols dataSet
####

pred_list = []

for trees in range(1, 26):
    for depth in range(1, 11):
        rf_clf_test = RandomForestClassifier(n_estimators = trees, 
                                    max_depth = depth, criterion ='entropy',
                                    random_state = 1337)
        rf_clf_test.fit(ols_df_train_x, ols_df_train_y)
        pred_list.append([trees, depth, 
                    round(np.mean(rf_clf_test.predict(ols_df_test_x) 
                    == ols_df_test_y) 
                    * 100, 2), 'ols'])

# create a dataframe from the classifer data
forest_df_3 = pd.DataFrame(pred_list, columns = ['Estimators', 'Depth',
                                                 'Accuracy', 'Set'])

# append forest 3 to full df
rf_accuracy_df = rf_accuracy_df.append(forest_df_3)

# store the lowest error rate value from the classifier
ind = forest_df_3.loc[forest_df_3['Accuracy'] == 
                      max(forest_df_3.Accuracy)].values
                      
# pull out the number of trees and depth
trees_depth.append(int(ind.item(0)))
trees_depth.append(int(ind.item(1)))

# append the models accruacy to the accuracy list
rf_accuracy.append(round(ind.item(2), 2))

print('OLS:\nOptimal trees:', trees_depth[4],
      '\nOptimal depth:', trees_depth[5])


####
# End ols dataSet
####

####
# Start rfe dataSet
####

pred_list = []

for trees in range(1, 26):
    for depth in range(1, 11):
        rf_clf_test = RandomForestClassifier(n_estimators = trees, 
                                    max_depth = depth, criterion ='entropy',
                                    random_state = 1337)
        rf_clf_test.fit(rfe_df_train_x, rfe_df_train_y)
        pred_list.append([trees, depth, 
                    round(np.mean(rf_clf_test.predict(rfe_df_test_x) 
                    == rfe_df_test_y) 
                    * 100, 2), 'rfe'])

# create a dataframe from the classifer data
forest_df_4 = pd.DataFrame(pred_list, columns = ['Estimators', 'Depth',
                                                 'Accuracy', 'Set'])

# append forest 4 to full df
rf_accuracy_df = rf_accuracy_df.append(forest_df_4)

# store the lowest error rate value from the classifier
ind = forest_df_4.loc[forest_df_4['Accuracy'] == 
                      max(forest_df_4.Accuracy)].values
                      
# pull out the number of trees and depth
trees_depth.append(int(ind.item(0)))
trees_depth.append(int(ind.item(1)))

# append the models accruacy to the accuracy list
rf_accuracy.append(round(ind.item(2), 2))

print('RFE:\nOptimal trees:', trees_depth[6],
      '\nOptimal depth:', trees_depth[7])

####
# End rfe dataSet
####

####
# Start lasso dataSet
####

pred_list = []

for trees in range(1, 26):
    for depth in range(1, 11):
        rf_clf_test = RandomForestClassifier(n_estimators = trees, 
                                    max_depth = depth, criterion ='entropy',
                                    random_state = 1337)
        rf_clf_test.fit(lasso_df_train_x, lasso_df_train_y)
        pred_list.append([trees, depth, 
                    round(np.mean(rf_clf_test.predict(lasso_df_test_x) 
                    == lasso_df_test_y) 
                    * 100, 2), 'lasso'])

# create a dataframe from the classifer data
forest_df_5 = pd.DataFrame(pred_list, columns = ['Estimators', 'Depth',
                                                 'Accuracy', 'Set'])

# append forest 5 to full df
rf_accuracy_df = rf_accuracy_df.append(forest_df_5)

# store the lowest error rate value from the classifier
ind = forest_df_5.loc[forest_df_5['Accuracy'] == 
                      max(forest_df_5.Accuracy)].values
                      
# pull out the number of trees and depth
trees_depth.append(int(ind.item(0)))
trees_depth.append(int(ind.item(1)))

# append the models accruacy to the accuracy list
rf_accuracy.append(round(ind.item(2), 2))

print('Lasso:\nOptimal trees:', trees_depth[8],
      '\nOptimal depth:', trees_depth[9])

####
# End lasso dataSet
####

####
# Start full dataSet
####

pred_list = []

for trees in range(1, 26):
    for depth in range(1, 11):
        rf_clf_test = RandomForestClassifier(n_estimators = trees, 
                                    max_depth = depth, criterion ='entropy',
                                    random_state = 1337)
        rf_clf_test.fit(full_df_train_x, full_df_train_y)
        pred_list.append([trees, depth, 
                    round(np.mean(rf_clf_test.predict(full_df_test_x) 
                    == full_df_test_y) 
                    * 100, 2), 'full'])

# create a dataframe from the classifer data
forest_df_6 = pd.DataFrame(pred_list, columns = ['Estimators', 'Depth',
                                                 'Accuracy', 'Set'])

# append forest 6 to full df
rf_accuracy_df = rf_accuracy_df.append(forest_df_6)
    
# store the lowest error rate value from the classifier
ind = forest_df_6.loc[forest_df_6['Accuracy'] == 
                      max(forest_df_6.Accuracy)].values
                      
# pull out the number of trees and depth
trees_depth.append(int(ind.item(0)))
trees_depth.append(int(ind.item(1)))

# append the models accruacy to the accuracy list
rf_accuracy.append(round(ind.item(2), 2))

print('Full:\nOptimal trees:', trees_depth[10],
      '\nOptimal depth:', trees_depth[11])

####
# End full dataSet
####

# Create palette
palette = dict(zip(rf_accuracy_df.Depth.unique(),
                   sns.color_palette("tab10", 10)))

# Plot
sns.relplot(x="Estimators", y="Accuracy",
            hue="Depth", col="Set",
            palette=palette, col_wrap=3,
            height=3, aspect=1, facet_kws=dict(sharex=False),
            kind="line", legend="full", data=rf_accuracy_df)


####
# End RF accuracy tests
####

####
# Start fixed value RFs
####

# start timing
timings_list.append(['Random forest duration:', time.time()])

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
rf_clf = RandomForestClassifier(n_estimators = 12, 
                                    max_depth = 9, criterion ='entropy',
                                    random_state = 1337)
rf_clf.fit(pear_five_df_train_x, pear_five_df_train_y)

pear_five_pred = rf_clf.predict(pear_five_df_test_x)

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
rf_clf = RandomForestClassifier(n_estimators = 24, 
                                    max_depth = 10, criterion ='entropy',
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
rf_clf = RandomForestClassifier(n_estimators = 22, 
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
rf_clf = RandomForestClassifier(n_estimators = 21, 
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
rf_clf = RandomForestClassifier(n_estimators = 24, 
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
rf_clf = RandomForestClassifier(n_estimators = 23, 
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

####
# End fixed value RFs
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
timings_list.append(['Scaled time', time.time()])
timings_list.append(['Knn duration:', time.time()])

#######
# Start KNN
#######

####
# Start pear-five dataset
####

# start time
algorithm_duration_list.append(time.time())

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

# start time
algorithm_duration_list.append(time.time())

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

# end time
algorithm_duration_list.append(time.time())

####
# end pear-ten dataset
####

####
# Start ols dataset
####

# start time
algorithm_duration_list.append(time.time())

# initalize knn
knn_classifier = KNeighborsClassifier(n_neighbors = 11)

# Train the classifier
knn_classifier.fit(ols_scaled_df_train_x, 
                   ols_scaled_df_train_y)
        
# store accuracy
global_accuracy.append(100-(round(np.mean(knn_classifier.predict(
                                          ols_scaled_df_test_x) 
                                          != ols_scaled_df_test_y),2)))

# end time
algorithm_duration_list.append(time.time()) 

####
# end ols dataset
####
    
####
# Start rfe dataset
####

# start time
algorithm_duration_list.append(time.time())

# initalize knn
knn_classifier = KNeighborsClassifier(n_neighbors = 15)

# Train the classifier
knn_classifier.fit(rfe_scaled_df_train_x, 
                   rfe_scaled_df_train_y)
        
# store accuracy
global_accuracy.append(100-(round(np.mean(knn_classifier.predict(
                                          rfe_scaled_df_test_x) 
                                          != rfe_scaled_df_test_y),2)))

# end time
algorithm_duration_list.append(time.time()) 

####
# end rfe dataset
####
    
####
# Start lasso dataset
####

# start time
algorithm_duration_list.append(time.time())

# initalize knn
knn_classifier = KNeighborsClassifier(n_neighbors = 25)

# Train the classifier
knn_classifier.fit(lasso_scaled_df_train_x, 
                   lasso_scaled_df_train_y)
        
# Perform predictions
pred_k = knn_classifier.predict(lasso_scaled_df_test_x)

###!!!!!WHY IS THIS IMPLEMENTED DIFFERENTLY ???!!!!
# Store accuracy
global_accuracy.append(round(sum(pred_k == lasso_scaled_df_test_y) 
                              / len(pred_k) * 100, 2))

# end time
algorithm_duration_list.append(time.time()) 

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
timings_list.append(['SVM linear duration:', time.time()])

####
# Start pear five dataset
####

# start time
algorithm_duration_list.append(time.time())

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

# end time
algorithm_duration_list.append(time.time())

####
# End pear ten dataset
####

####
# Start ols ten dataset
####

# start time
algorithm_duration_list.append(time.time())

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

# end time
algorithm_duration_list.append(time.time())

####
# End ols ten dataset
####

####
# Start rfe ten dataset
####

# start time
algorithm_duration_list.append(time.time())

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

# end time
algorithm_duration_list.append(time.time())

####
# End rfe ten dataset
####

####
# Start lasso ten dataset
####

# start time
algorithm_duration_list.append(time.time())

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

# end time
algorithm_duration_list.append(time.time())

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
timings_list.append(['SVM Gaussian duration:', time.time()])

####
# Start pear five dataset
####

# start time
algorithm_duration_list.append(time.time())

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

# end time
algorithm_duration_list.append(time.time())

####
# End lasso dataset
####

#######
# End rbf SVM
#######

# end timing
timings_list.append(['svm rbf time end', time.time()])

#######
# Start poly SVM
#######

# start timing
timings_list.append(['SVM poly duration:', time.time()])

####
# Start pear five dataset
####

# start time
algorithm_duration_list.append(time.time())

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

# end time
algorithm_duration_list.append(time.time())

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
timings_list.append(['Logistic Regression liblinear duration:', time.time()])

####
# Start pear five dataset
####

# start time
algorithm_duration_list.append(time.time())

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

# end time
algorithm_duration_list.append(time.time())

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
timings_list.append(['Logistic Regression sag duration:', time.time()])

####
# Start pear five dataset
####

# start time
algorithm_duration_list.append(time.time())

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

# end time
algorithm_duration_list.append(time.time())

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
timings_list.append(['Logistic Regression newton-cg duration:', time.time()])

####
# Start pear five dataset
####

# start time
algorithm_duration_list.append(time.time())

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

# end time
algorithm_duration_list.append(time.time())

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
timings_list.append(['global time end', time.time()])

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
                                       'accuracy', 'time'])

# Build out a dataframe to store the classifiers and their accuracy
for i in range(0, len(classifiers)):
    for k in range(0, len(attributes)):
        prediction_df = prediction_df.append({'classifier' : classifiers[i],
                                      'details' : 'None',
                                      'attributes' : attributes[k],
                                      'accuracy' : 0,
                                      'time' : 0}, 
                                      ignore_index=True)

# Move indexes down 1 starting at 15 to add Random forest with full dataset
prediction_df.index = (prediction_df.index[:15].tolist() + 
                      (prediction_df.index[15:] + 1).tolist())

# Adds Random forest with full dataset to the dataframe
prediction_df.loc[15] = ['Random Forest', 'None', 'Full', 0, 0]

# reorders the indexes after the insert
prediction_df = prediction_df.sort_index()

# Updates the accuracy
prediction_df['accuracy'] = global_accuracy

# Manually set algorithm details
# decision tree
prediction_df['details'].iloc[0:5] = 'Entropy'

# Naive Bayes
prediction_df['details'].iloc[5:11] = 'Gaussian'

# Random Forest tree/depth
prediction_df['details'].iloc[10] = '8 trees, 8 depth'
prediction_df['details'].iloc[11] = '23 trees, 9 depth'
prediction_df['details'].iloc[12] = '25 trees, 10 depth'
prediction_df['details'].iloc[13] = '20 trees, 9 depth'
prediction_df['details'].iloc[14] = '22 trees, 10 depth'
prediction_df['details'].iloc[15] = '25 trees, 10 depth'
    
# knn k value
prediction_df['details'].iloc[16] = 'K - 15'
prediction_df['details'].iloc[17] = 'K - 9'
prediction_df['details'].iloc[18] = 'K - 11'
prediction_df['details'].iloc[19] = 'K - 15'
prediction_df['details'].iloc[20] = 'K - 25'

# log liblinear solver
prediction_df['details'].iloc[21:26] = 'Linear'

# log sag solver
prediction_df['details'].iloc[26:31] = 'Gaussian'

# log newton-cg solver
prediction_df['details'].iloc[31:36] = 'Poly'

# log liblinear solver
prediction_df['details'].iloc[36:41] = 'Liblinear'

# log sag solver
prediction_df['details'].iloc[41:46] = 'Sag'

# log newton-cg solver
prediction_df['details'].iloc[46:51] = 'Newton-cg'

# creates a duration list
dur_list = []

# stores the durations of each algorithm from start to finish
for dur in range(0, len(algorithm_duration_list), 2):
    dur_list.append(algorithm_duration_list[dur + 1] - 
                    algorithm_duration_list[dur])

# stores the durations into the prediction_df
prediction_df['time'] = dur_list

# Print durations 
print('Durations for each of the major steps in the process,', 
      'measured in seconds\n', timings_list[1][0],
int(timings_list[2][1]) - int(timings_list[1][1]),'\n',

timings_list[3][0],
int(timings_list[4][1]) - int(timings_list[3][1]),'\n',

timings_list[5][0],
int(timings_list[6][1]) - int(timings_list[5][1]),'\n',

timings_list[8][0],
int(timings_list[9][1]) - int(timings_list[8][1]),'\n',

timings_list[10][0],
int(timings_list[11][1]) - int(timings_list[10][1]),'\n',

timings_list[12][0],
int(timings_list[13][1]) - int(timings_list[12][1]),'\n',

timings_list[7][0],
int(timings_list[14][1]) - int(timings_list[7][1]),'\n',

timings_list[16][0],
int(timings_list[17][1]) - int(timings_list[16][1]),'\n',

timings_list[18][0],
int(timings_list[19][1]) - int(timings_list[18][1]),'\n',

timings_list[20][0],
int(timings_list[21][1]) - int(timings_list[20][1]),'\n',

timings_list[22][0],
int(timings_list[23][1]) - int(timings_list[22][1]),'\n',

timings_list[24][0],
int(timings_list[25][1]) - int(timings_list[24][1]),'\n',

timings_list[26][0],
int(timings_list[27][1]) - int(timings_list[26][1]),'\n',

timings_list[28][0],
int(timings_list[29][1]) - int(timings_list[28][1]),'\n',

timings_list[15][0],
int(timings_list[30][1]) - int(timings_list[15][1]),'\n',

timings_list[0][0],
int(timings_list[-1][1])- int(timings_list[0][1]))


# Finds the most and least accurate algorithms
accurate = [prediction_df[prediction_df['accuracy'] == 
              max(prediction_df.accuracy)].values[0]]

least_accurate = [prediction_df[prediction_df['accuracy'] == 
              min(prediction_df.accuracy)].values[0]]


print('The most accurate classifier was', accurate[0][0], 'using', 
      accurate[0][1], 'on the', accurate[0][2], 
      'attribute set with an accuracy of', accurate[0][3],'%')

print('The least accurate classifier was', least_accurate[0][0], 'using', 
      least_accurate[0][1], 'on the', least_accurate[0][2], 
      'attribute set with an accuracy of', least_accurate[0][3],'%')

# confusion matrix for random forest 8/8
cm_one = confusion_matrix(pear_five_df_test_y, pear_five_pred)
tn, fp, fn, tp  = confusion_matrix(pear_five_df_test_y, pear_five_pred).ravel()

# Create confusion matrix heatmap
# setup class names and tick marks
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap and labels
sns.heatmap(pd.DataFrame(cm_one), annot=True, cmap="summer" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for Random Forest', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# TPR/TNR rates
print('The TPR is:', str(tp) + '/' + str(tp + fn) + ',',
      round(recall_score(pear_five_df_test_y, pear_five_pred) * 100, 2),'%')
print('The TNR is:', str(tn) + '/' + str(tn + fp) + ',',
    round(tn / (tn + fp) * 100, 2),'%')

####
# End prediction prints
####