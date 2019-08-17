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
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import confusion_matrix, recall_score

# set seaborn to dark backgrounds
sns.set_style("darkgrid")

# setup input directory and filename
data = 'games'
input_dir = r'C:\Users\TomBrody\Desktop\School\677\Final'
csv_file = os.path.join(input_dir, data + '.csv')

# read csv file into dataframe
try:
    lol_df = pd.read_csv(csv_file)
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
lol_x = lol_df.drop('win', 1)

# y stores only the win column since its used as a predictor
lol_y = lol_df['win']

################
# Start attribute selection with various methods
################

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

# removed win column
# create dataframe top 5 correlated attributes
pear_five_df = lol_df[['firstInhibitor', 't1_towerKills', 't1_inhibitorKills', 
                      't2_towerKills', 't2_inhibitorKills']] #,'win']]

# removed win column
# create dataframe top 10 correlated attributes
pear_ten_df = lol_df[['firstTower','firstInhibitor', 't1_towerKills',
                       't1_inhibitorKills', 't1_baronKills', 't1_dragonKills',
                       't2_towerKills', 't2_inhibitorKills', 't2_baronKills',
                       't2_dragonKills']] #, 'win']]

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
    p = pd.Series(model.pvalues.values[1:], index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax > 0.05):
        cols.remove(feature_with_p_max)
    else:
        break

# sets and prints the remaining unremoved features
selected_features_BE = cols
print(selected_features_BE)

# removed win column
# Appends win column for prediction purposes
#selected_features_BE.append('win')

# creates a dataframe with the ols selected columns
ols_df = lol_df[selected_features_BE]

########
# End Ordinary Least Squares
########

########
# Start Recursive Feature Elimination
########

######!!!!!!!!!
# only used to determine optimum number of attributes
######

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

######!!!!!!!!!
# doesnt have to be run every time
######

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

# removed win column
# create index object for win column
#win_df = pd.Index(['win'])

# Appends win column for prediction purposes
# selected_features_rfe = selected_features_rfe.append([win_df])

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

# removed win column add
#lasso_df['win'] = lol_df.win

########
# End lasso method
########

################
# End attribute selection with various methods
################

################
# Start building scaled dataframes
################

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
pear_five_df
pear_ten_df
ols_df
rfe_df
lasso_df

# pear_five split dataset into 33% test 66% training
pear_five_df_train_x, pear_five_df_test_x, pear_five_df_train_y, pear_five_df_test_y = (
        train_test_split(pear_five_df, lol_y, test_size = 0.33, 
                         random_state=1337))

#pear_five_df_train_x
#pear_five_df_test_x
#pear_five_df_train_y
#pear_five_df_test_y

# pear_ten split dataset into 33% test 66% training
pear_ten_df_train_x, pear_ten_df_test_x, pear_ten_df_train_y, pear_ten_df_test_y = (
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

################
# End building test/train datasets with various attribute selections
################

################
# Start building non-scaled algorithms
################

#######
# Start decision tree
#######

# Create a decisions tree classifier
pear_five_tree_clf = tree.DecisionTreeClassifier(criterion = 'entropy')

# Train the classifier on pearsons top 5 attributes
pear_five_tree_clf = pear_five_tree_clf.fit(pear_five_df_train_x, 
                                            pear_five_df_train_y)

# Predict on pearsons top 5 attributes
pear_five_prediction = pear_five_tree_clf.predict(pear_five_df_test_x)

# calculate error rate
pear_five_tree_acc = 100-(round(np.mean(pear_five_prediction 
                                             != pear_five_df_test_y) * 100, 2))

####
# End five
####

####
# Start ten
####

# Create a decisions tree classifier
pear_ten_tree_clf = tree.DecisionTreeClassifier(criterion = 'entropy')

# Train the classifier on pearsons top 10 attributes
pear_ten_tree_clf = pear_ten_tree_clf.fit(pear_ten_df_train_x, 
                                            pear_ten_df_train_y)

# Predict on pearsons top 10 attributes
pear_ten_prediction = pear_ten_tree_clf.predict(pear_ten_df_test_x)

# calculate error rate
pear_ten_tree_acc = 100-(round(np.mean(pear_ten_prediction 
                                             != pear_ten_df_test_y) * 100, 2))

####
# End ten
####

####
# Start ols_df
####

# Create a decisions tree classifier
ols_df_tree_clf = tree.DecisionTreeClassifier(criterion = 'entropy')

# Train the classifier on ols attributes
ols_df_tree_clf = ols_df_tree_clf.fit(ols_df_train_x, 
                                            ols_df_train_y)

# Predict on ols attributes
ols_df_prediction = ols_df_tree_clf.predict(ols_df_test_x)

# calculate error rate
ols_df_tree_acc= 100-(round(np.mean(ols_df_prediction 
                                             != ols_df_test_y) * 100, 2))

####
# End ols_df
####

####
# Start rfe_df
####

# Create a decisions tree classifier
rfe_df_tree_clf = tree.DecisionTreeClassifier(criterion = 'entropy')

# Train the classifier on rfe attributes
rfe_df_tree_clf = rfe_df_tree_clf.fit(rfe_df_train_x, 
                                            rfe_df_train_y)

# Predict on rfe attributes
rfe_df_prediction = rfe_df_tree_clf.predict(rfe_df_test_x)

# calculate error rate
rfe_df_tree_acc = 100-(round(np.mean(rfe_df_prediction 
                                             != rfe_df_test_y) * 100, 2))

####
# End rfe_df
####

####
# Start lasso_df
####

# Create a decisions tree classifier
lasso_df_tree_clf = tree.DecisionTreeClassifier(criterion = 'entropy')

# Train the classifier on lasso attributes
lasso_df_tree_clf = lasso_df_tree_clf.fit(lasso_df_train_x, 
                                            lasso_df_train_y)

# Predict on lasso attributes
lasso_df_prediction = lasso_df_tree_clf.predict(lasso_df_test_x)

# calculate error rate
lasso_df_tree_acc = 100-(round(np.mean(lasso_df_prediction 
                                             != lasso_df_test_y) * 100, 2))

####
# End lasso_df
####

####
# Start prediction prints
####
print('pear_five_tree_acc:', pear_five_tree_acc)
print('pear_ten_tree_acc:', pear_ten_tree_acc)
print('ols_df_tree_acc:', ols_df_tree_acc)
print('rfe_df_tree_acc:', rfe_df_tree_acc)
print('lasso_df_tree_acc:', lasso_df_tree_acc)
####
# End prediction prints
####

#######
# End decision tree
#######

#######
# Start naive bayes
#######

# Create a naive bayes classifier
pear_five_gnb_clf = GaussianNB()

# Train the classifier on pearsons top 5 attributes
pear_five_gnb_clf = pear_five_gnb_clf.fit(pear_five_df_train_x, 
                                            pear_five_df_train_y)

# Predict on pearsons top 5 attributes
pear_five_prediction = pear_five_gnb_clf.predict(pear_five_df_test_x)

# calculate error rate
pear_five_gnb_acc = 100-(round(np.mean(pear_five_prediction 
                                             != pear_five_df_test_y) * 100, 2))

####
# End five
####

####
# Start ten
####

# Create a naive bayes classifier
pear_ten_gnb_clf = GaussianNB()

# Train the classifier on pearsons top 10 attributes
pear_ten_gnb_clf = pear_ten_gnb_clf.fit(pear_ten_df_train_x, 
                                            pear_ten_df_train_y)

# Predict on pearsons top 10 attributes
pear_ten_prediction = pear_ten_gnb_clf.predict(pear_ten_df_test_x)

# calculate error rate
pear_ten_gnb_acc = 100-(round(np.mean(pear_ten_prediction 
                                             != pear_ten_df_test_y) * 100, 2))

####
# End ten
####

####
# Start ols_df
####

# Create a naive bayes classifier
ols_df_gnb_clf = GaussianNB()

# Train the classifier on ols attributes
ols_df_gnb_clf = ols_df_gnb_clf.fit(ols_df_train_x, 
                                            ols_df_train_y)

# Predict on ols attributes
ols_df_prediction = ols_df_gnb_clf.predict(ols_df_test_x)

# calculate error rate
ols_df_gnb_acc = 100-(round(np.mean(ols_df_prediction 
                                             != ols_df_test_y) * 100, 2))

####
# End ols_df
####

####
# Start rfe_df
####

# Create a naive bayes classifier
rfe_df_gnb_clf = GaussianNB()

# Train the classifier on rfe attributes
rfe_df_gnb_clf = rfe_df_gnb_clf.fit(rfe_df_train_x, 
                                            rfe_df_train_y)

# Predict on rfe attributes
rfe_df_prediction = rfe_df_gnb_clf.predict(rfe_df_test_x)

# calculate error rate
rfe_df_gnb_acc = 100-(round(np.mean(rfe_df_prediction 
                                             != rfe_df_test_y) * 100, 2))

####
# End rfe_df
####

####
# Start lasso_df
####

# Create a naive bayes classifier
lasso_df_gnb_clf = GaussianNB()

# Train the classifier on lasso attributes
lasso_df_gnb_clf = lasso_df_gnb_clf.fit(lasso_df_train_x, 
                                            lasso_df_train_y)

# Predict on lasso attributes
lasso_df_prediction = lasso_df_gnb_clf.predict(lasso_df_test_x)

# calculate error rate
lasso_df_gnb_acc = 100-(round(np.mean(lasso_df_prediction 
                                             != lasso_df_test_y) * 100, 2))

####
# End lasso_df
####

####
# Start prediction prints
####
print('pear_five_gnb_acc:', pear_five_gnb_acc)
print('pear_ten_gnb_acc:', pear_ten_gnb_acc)
print('ols_df_gnb_acc:', ols_df_gnb_acc)
print('rfe_df_gnb_acc:', rfe_df_gnb_acc)
print('lasso_df_gnb_acc:', lasso_df_gnb_acc)
####
# End prediction prints
####

#######
# End naive bayes
#######

################
# End building non-scaled algorithms
################

################
# Start building scaled algorithms
################

#######
# Start KNN
#######

####
# Start pear-five dataset
####

current_set = 'pear-five'

# Create empty lists to store the models error rate and 
# accuracy across various K's
error_rate = []
accuracy = []
k_value = []

# For loop to test the model using various neighbor sizes
# with k neighbors set to 3 to 25
try:
    for k in range (3, 27, 2):
        # Create the classifier with neighbors set to k from the loop
        knn_classifier = KNeighborsClassifier(n_neighbors = k)
       
        # Train the classifier
        knn_classifier.fit(pear_five_scaled_df_train_x, 
                           pear_five_scaled_df_train_y)
        
        # Perform predictions
        pred_k = knn_classifier.predict(pear_five_scaled_df_test_x)
        
        # Store error rate and accuracy for particular K value
        k_value.append(k)
        error_rate.append(round(np.mean(
                pred_k != pear_five_scaled_df_test_y) * 100, 2))
        
        accuracy.append(round(sum(
                pred_k == pear_five_scaled_df_test_y) / len(pred_k) * 100, 2))
        
except Exception as e:
    print(e)
    print('failed to build the KNN classifier.')

for i in range (0,12):
    print('The accuracy on the pearson five data when K =', k_value[i], 
          'is:', accuracy[i])
    
# create a plot to display the accuracy of the model across K
fig = plt.figure(figsize=(10, 4))
ax = plt.gca()
plt.plot(range(3, 27, 2), accuracy, color ='blue',
         marker = 'o', markerfacecolor = 'black', markersize = 10)
plt.title('Accuracy vs. k for the pearson five dataset')
plt.xlabel('Number of neighbors: k')
plt.ylabel('Accuracy')

max_index = accuracy.index(max(accuracy))

highest_accuracy = accuracy[max_index]
best_set = 'pearson five'

print('The most accurate k for the pearson five attribute selection is', 
      k_value[max_index], 'at', accuracy[max_index],'%')

####
# end pear-five dataset
####

####
# Start pear-ten dataset
####

current_set = 'pear-ten'

# Create empty lists to store the models error rate and 
# accuracy across various K's
error_rate = []
accuracy = []
k_value = []

# For loop to test the model using various neighbor sizes
# with k neighbors set to 3 to 25
try:
    for k in range (3, 27, 2):
        # Create the classifier with neighbors set to k from the loop
        knn_classifier = KNeighborsClassifier(n_neighbors = k)
       
        # Train the classifier
        knn_classifier.fit(pear_ten_scaled_df_train_x, 
                           pear_ten_scaled_df_train_y)
        
        # Perform predictions
        pred_k = knn_classifier.predict(pear_ten_scaled_df_test_x)
        
        # Store error rate and accuracy for particular K value
        k_value.append(k)
        error_rate.append(round(np.mean(
                pred_k != pear_ten_scaled_df_test_y) * 100, 2))
        
        accuracy.append(round(sum(
                pred_k == pear_ten_scaled_df_test_y) / len(pred_k) * 100, 2))
        
except Exception as e:
    print(e)
    print('failed to build the KNN classifier.')

for i in range (0,12):
    print('The accuracy on the pearson ten data when K =', k_value[i], 
          'is:', accuracy[i])
    
# create a plot to display the accuracy of the model across K
fig = plt.figure(figsize=(10, 4))
ax = plt.gca()
plt.plot(range(3, 27, 2), accuracy, color ='blue',
         marker = 'o', markerfacecolor = 'black', markersize = 10)
plt.title('Accuracy vs. k for the pearson ten dataset')
plt.xlabel('Number of neighbors: k')
plt.ylabel('Accuracy')

max_index = accuracy.index(max(accuracy))

print('The most accurate k for the pearson ten attribute selection is', 
      k_value[max_index], 'at', accuracy[max_index],'%')

if accuracy[max_index] > highest_accuracy:
    highest_accuracy = accuracy[max_index]
    best_set = current_set

####
# end pear-ten dataset
####

####
# Start ols dataset
####

current_set = 'ols'

# Create empty lists to store the models error rate and 
# accuracy across various K's
error_rate = []
accuracy = []
k_value = []

# For loop to test the model using various neighbor sizes
# with k neighbors set to 3 to 25
try:
    for k in range (3, 27, 2):
        # Create the classifier with neighbors set to k from the loop
        knn_classifier = KNeighborsClassifier(n_neighbors = k)
       
        # Train the classifier
        knn_classifier.fit(ols_scaled_df_train_x, 
                           ols_scaled_df_train_y)
        
        # Perform predictions
        pred_k = knn_classifier.predict(ols_scaled_df_test_x)
        
        # Store error rate and accuracy for particular K value
        k_value.append(k)
        error_rate.append(round(np.mean(
                pred_k != ols_scaled_df_test_y) * 100, 2))
        
        accuracy.append(round(sum(
                pred_k == ols_scaled_df_test_y) / len(pred_k) * 100, 2))
        
except Exception as e:
    print(e)
    print('failed to build the KNN classifier.')

for i in range (0,12):
    print('The accuracy on the ols data when K =', k_value[i], 
          'is:', accuracy[i])
    
# create a plot to display the accuracy of the model across K
fig = plt.figure(figsize=(10, 4))
ax = plt.gca()
plt.plot(range(3, 27, 2), accuracy, color ='blue',
         marker = 'o', markerfacecolor = 'black', markersize = 10)
plt.title('Accuracy vs. k for the ols dataset')
plt.xlabel('Number of neighbors: k')
plt.ylabel('Accuracy')

max_index = accuracy.index(max(accuracy))

print('The most accurate k for the ols attribute selection is', 
      k_value[max_index], 'at', accuracy[max_index],'%')

if accuracy[max_index] > highest_accuracy:
    highest_accuracy = accuracy[max_index]
    best_set = current_set

####
# end ols dataset
####
    
####
# Start rfe dataset
####

current_set = 'rfe'

# Create empty lists to store the models error rate and 
# accuracy across various K's
error_rate = []
accuracy = []
k_value = []

# For loop to test the model using various neighbor sizes
# with k neighbors set to 3 to 25
try:
    for k in range (3, 27, 2):
        # Create the classifier with neighbors set to k from the loop
        knn_classifier = KNeighborsClassifier(n_neighbors = k)
       
        # Train the classifier
        knn_classifier.fit(rfe_scaled_df_train_x, 
                           rfe_scaled_df_train_y)
        
        # Perform predictions
        pred_k = knn_classifier.predict(rfe_scaled_df_test_x)
        
        # Store error rate and accuracy for particular K value
        k_value.append(k)
        error_rate.append(round(np.mean(
                pred_k != rfe_scaled_df_test_y) * 100, 2))
        
        accuracy.append(round(sum(
                pred_k == rfe_scaled_df_test_y) / len(pred_k) * 100, 2))
        
except Exception as e:
    print(e)
    print('failed to build the KNN classifier.')

for i in range (0,12):
    print('The accuracy on the rfe data when K =', k_value[i], 
          'is:', accuracy[i])
    
# create a plot to display the accuracy of the model across K
fig = plt.figure(figsize=(10, 4))
ax = plt.gca()
plt.plot(range(3, 27, 2), accuracy, color ='blue',
         marker = 'o', markerfacecolor = 'black', markersize = 10)
plt.title('Accuracy vs. k for the rfe dataset')
plt.xlabel('Number of neighbors: k')
plt.ylabel('Accuracy')

max_index = accuracy.index(max(accuracy))

print('The most accurate k for the rfe attribute selection is', 
      k_value[max_index], 'at', accuracy[max_index],'%')

if accuracy[max_index] > highest_accuracy:
    highest_accuracy = accuracy[max_index]
    best_set = current_set

####
# end rfe dataset
####
    
####
# Start lasso dataset
####

current_set = 'lasso'

# Create empty lists to store the models error rate and 
# accuracy across various K's
error_rate = []
accuracy = []
k_value = []

# For loop to test the model using various neighbor sizes
# with k neighbors set to 3 to 25
try:
    for k in range (3, 27, 2):
        # Create the classifier with neighbors set to k from the loop
        knn_classifier = KNeighborsClassifier(n_neighbors = k)
       
        # Train the classifier
        knn_classifier.fit(lasso_scaled_df_train_x, 
                           lasso_scaled_df_train_y)
        
        # Perform predictions
        pred_k = knn_classifier.predict(lasso_scaled_df_test_x)
        
        # Store error rate and accuracy for particular K value
        k_value.append(k)
        error_rate.append(round(np.mean(
                pred_k != lasso_scaled_df_test_y) * 100, 2))
        
        accuracy.append(round(sum(
                pred_k == lasso_scaled_df_test_y) / len(pred_k) * 100, 2))
        
except Exception as e:
    print(e)
    print('failed to build the KNN classifier.')

for i in range (0,12):
    print('The accuracy on the lasso data when K =', k_value[i], 
          'is:', accuracy[i])
    
# create a plot to display the accuracy of the model across K
fig = plt.figure(figsize=(10, 4))
ax = plt.gca()
plt.plot(range(3, 27, 2), accuracy, color ='blue',
         marker = 'o', markerfacecolor = 'black', markersize = 10)
plt.title('Accuracy vs. k for the lasso dataset')
plt.xlabel('Number of neighbors: k')
plt.ylabel('Accuracy')

max_index = accuracy.index(max(accuracy))

print('The most accurate k for the lasso attribute selection is', 
      k_value[max_index], 'at', accuracy[max_index],'%')

if accuracy[max_index] > highest_accuracy:
    highest_accuracy = accuracy[max_index]
    best_set = current_set

####
# end lasso dataset
####

print(highest_accuracy, best_set)

#######
# End KNN
#######

#######
# Start linear SVM
#######

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
accuracy_rate_linear = 100-(round(np.mean(prediction_linear != 
                                         pear_five_scaled_df_test_y) * 100, 2))

print('The linear SVM classifier has an accuracy of', accuracy_rate_linear,'%',
      'on the pearson five attribute set')

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
accuracy_rate_linear = 100-(round(np.mean(prediction_linear != 
                                         pear_ten_scaled_df_test_y) * 100, 2))

print('The linear SVM classifier has an accuracy of', accuracy_rate_linear,'%',
      'on the pearson ten attribute set')
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
accuracy_rate_linear = 100-(round(np.mean(prediction_linear != 
                                         ols_scaled_df_test_y) * 100, 2))

print('The linear SVM classifier has an accuracy of', accuracy_rate_linear,'%',
      'on the ols attribute set')
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
accuracy_rate_linear = 100-(round(np.mean(prediction_linear != 
                                         rfe_scaled_df_test_y) * 100, 2))

print('The linear SVM classifier has an accuracy of', accuracy_rate_linear,'%',
      'on the rfe attribute set')
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
accuracy_rate_linear = 100-(round(np.mean(prediction_linear != 
                                         lasso_scaled_df_test_y) * 100, 2))

print('The linear SVM classifier has an accuracy of', accuracy_rate_linear,'%',
      'on the lasso attribute set')
####
# End lasso ten dataset
####

#######
# End linear SVM
#######

#######
# Start rbf SVM
#######

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
accuracy_rate_rbf = 100-(round(np.mean(prediction_rbf != 
                                         pear_five_scaled_df_test_y) * 100, 2))

print('The rbf SVM classifier has an accuracy of', accuracy_rate_rbf,'%',
      'on the pearson five attribute set')

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
accuracy_rate_rbf = 100-(round(np.mean(prediction_rbf != 
                                         pear_ten_scaled_df_test_y) * 100, 2))

print('The rbf SVM classifier has an accuracy of', accuracy_rate_rbf,'%',
      'on the pearson ten attribute set')
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
accuracy_rate_rbf = 100-(round(np.mean(prediction_rbf != 
                                         ols_scaled_df_test_y) * 100, 2))

print('The rbf SVM classifier has an accuracy of', accuracy_rate_rbf,'%',
      'on the ols attribute set')
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
accuracy_rate_rbf = 100-(round(np.mean(prediction_rbf != 
                                         rfe_scaled_df_test_y) * 100, 2))

print('The rbf SVM classifier has an accuracy of', accuracy_rate_rbf,'%',
      'on the rfe attribute set')
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
accuracy_rate_rbf = 100-(round(np.mean(prediction_rbf != 
                                         lasso_scaled_df_test_y) * 100, 2))

print('The rbf SVM classifier has an accuracy of', accuracy_rate_rbf,'%',
      'on the lasso attribute set')
####
# End lasso ten dataset
####

#######
# End rbf SVM
#######

#######
# Start poly SVM
#######

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
accuracy_rate_poly = 100-(round(np.mean(prediction_poly != 
                                         pear_five_scaled_df_test_y) * 100, 2))

print('The poly SVM classifier has an accuracy of', accuracy_rate_poly,'%',
      'on the pearson five attribute set')

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
accuracy_rate_poly = 100-(round(np.mean(prediction_poly != 
                                         pear_ten_scaled_df_test_y) * 100, 2))

print('The poly SVM classifier has an accuracy of', accuracy_rate_poly,'%',
      'on the pearson ten attribute set')
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
accuracy_rate_poly = 100-(round(np.mean(prediction_poly != 
                                         ols_scaled_df_test_y) * 100, 2))

print('The poly SVM classifier has an accuracy of', accuracy_rate_poly,'%',
      'on the ols attribute set')
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
accuracy_rate_poly = 100-(round(np.mean(prediction_poly != 
                                         rfe_scaled_df_test_y) * 100, 2))

print('The poly SVM classifier has an accuracy of', accuracy_rate_poly,'%',
      'on the rfe attribute set')
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
accuracy_rate_poly = 100-(round(np.mean(prediction_poly != 
                                         lasso_scaled_df_test_y) * 100, 2))

print('The poly SVM classifier has an accuracy of', accuracy_rate_poly,'%',
      'on the lasso attribute set')
####
# End lasso dataset
####

#######
# End poly SVM
#######

#############
# End SVM
#############

#############
# Start Random Forest
#############

####
# Start pear five dataset
####

pred_list = []

# Create a random forest classifier that trains and predicts labels
# using from 1 to 25 trees and from 1 to 10 depth of each tree
# set random state to 1337 for repeatability
for trees in range(1, 26):
    for depth in range(1, 11):
        rf_clf = RandomForestClassifier(n_estimators = trees, 
                                    max_depth = depth, criterion ='entropy',
                                    random_state = 1337)
        rf_clf.fit(pear_five_scaled_df_train_x, pear_five_scaled_df_train_y)
        pred_list.append([trees, depth, 
                    round(np.mean(rf_clf.predict(pear_five_scaled_df_test_x) 
                    != pear_five_scaled_df_test_y) 
                    * 100, 2)])

# create a dataframe from the classifer data
forest_df = pd.DataFrame(pred_list, columns = ['estimators', 'depth', 'error_rate'])

# Plot the classifier data
sns.lineplot(x='estimators', y='error_rate', hue='depth', data=forest_df, palette="tab10",
             legend='full', ci = None)
plt.title('Random Forest error rates using pearson five attributes')
plt.ylabel('Error Rate')
plt.xlabel('Estimators')
plt.show()

print(forest_df.loc[forest_df['error_rate'] == min(forest_df.error_rate)])

####
# End pear five dataset
####

####
# Start pear ten dataset
####

pred_list = []

# Create a random forest classifier that trains and predicts labels
# using from 1 to 25 trees and from 1 to 10 depth of each tree
# set random state to 1337 for repeatability
for trees in range(1, 26):
    for depth in range(1, 11):
        rf_clf = RandomForestClassifier(n_estimators = trees, 
                                    max_depth = depth, criterion ='entropy',
                                    random_state = 1337)
        rf_clf.fit(pear_ten_scaled_df_train_x, pear_ten_scaled_df_train_y)
        pred_list.append([trees, depth, 
                    round(np.mean(rf_clf.predict(pear_ten_scaled_df_test_x) 
                    != pear_ten_scaled_df_test_y) 
                    * 100, 2)])

# create a dataframe from the classifer data
forest_df = pd.DataFrame(pred_list, columns = ['estimators', 'depth', 'error_rate'])

# Plot the classifier data
sns.lineplot(x='estimators', y='error_rate', hue='depth', data=forest_df, palette="tab10",
             legend='full', ci = None)
plt.title('Random Forest error rates using pearson ten attributes')
plt.ylabel('Error Rate')
plt.xlabel('Estimators')
plt.show()

print(forest_df.loc[forest_df['error_rate'] == min(forest_df.error_rate)])

####
# End pear ten dataset
####

####
# Start ols dataset
####

pred_list = []

# Create a random forest classifier that trains and predicts labels
# using from 1 to 25 trees and from 1 to 10 depth of each tree
# set random state to 1337 for repeatability
for trees in range(1, 26):
    for depth in range(1, 11):
        rf_clf = RandomForestClassifier(n_estimators = trees, 
                                    max_depth = depth, criterion ='entropy',
                                    random_state = 1337)
        rf_clf.fit(ols_scaled_df_train_x, ols_scaled_df_train_y)
        pred_list.append([trees, depth, 
                    round(np.mean(rf_clf.predict(ols_scaled_df_test_x) 
                    != ols_scaled_df_test_y) 
                    * 100, 2)])

# create a dataframe from the classifer data
forest_df = pd.DataFrame(pred_list, columns = ['estimators', 'depth', 'error_rate'])

# Plot the classifier data
sns.lineplot(x='estimators', y='error_rate', hue='depth', data=forest_df, palette="tab10",
             legend='full', ci = None)
plt.title('Random Forest error rates using ols attributes')
plt.ylabel('Error Rate')
plt.xlabel('Estimators')
plt.show()

print(forest_df.loc[forest_df['error_rate'] == min(forest_df.error_rate)])

####
# End ols dataset
####

####
# Start rfe dataset
####

pred_list = []

# Create a random forest classifier that trains and predicts labels
# using from 1 to 25 trees and from 1 to 10 depth of each tree
# set random state to 1337 for repeatability
for trees in range(1, 26):
    for depth in range(1, 11):
        rf_clf = RandomForestClassifier(n_estimators = trees, 
                                    max_depth = depth, criterion ='entropy',
                                    random_state = 1337)
        rf_clf.fit(rfe_scaled_df_train_x, rfe_scaled_df_train_y)
        pred_list.append([trees, depth, 
                    round(np.mean(rf_clf.predict(rfe_scaled_df_test_x) 
                    != rfe_scaled_df_test_y) 
                    * 100, 2)])

# create a dataframe from the classifer data
forest_df = pd.DataFrame(pred_list, columns = ['estimators', 'depth', 'error_rate'])

# Plot the classifier data
sns.lineplot(x='estimators', y='error_rate', hue='depth', data=forest_df, palette="tab10",
             legend='full', ci = None)
plt.title('Random Forest error rates using rfe attributes')
plt.ylabel('Error Rate')
plt.xlabel('Estimators')
plt.show()

print(forest_df.loc[forest_df['error_rate'] == min(forest_df.error_rate)])

####
# End rfe dataset
####

####
# Start lasso dataset
####

pred_list = []

# Create a random forest classifier that trains and predicts labels
# using from 1 to 25 trees and from 1 to 10 depth of each tree
# set random state to 1337 for repeatability
for trees in range(1, 26):
    for depth in range(1, 11):
        rf_clf = RandomForestClassifier(n_estimators = trees, 
                                    max_depth = depth, criterion ='entropy',
                                    random_state = 1337)
        rf_clf.fit(lasso_scaled_df_train_x, lasso_scaled_df_train_y)
        pred_list.append([trees, depth, 
                    round(np.mean(rf_clf.predict(lasso_scaled_df_train_y) 
                    != rfe_scaled_df_test_y) 
                    * 100, 2)])

# create a dataframe from the classifer data
forest_df = pd.DataFrame(pred_list, columns = ['estimators', 'depth', 'error_rate'])

# Plot the classifier data
sns.lineplot(x='estimators', y='error_rate', hue='depth', data=forest_df, palette="tab10",
             legend='full', ci = None)
plt.title('Random Forest error rates using rfe attributes')
plt.ylabel('Error Rate')
plt.xlabel('Estimators')
plt.show()

print(forest_df.loc[forest_df['error_rate'] == min(forest_df.error_rate)])

####
# End lasso dataset
####

#############
# End Random Forest
#############

#############
# Start log regression liblinear solver
#############

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

accuracy = np.mean(prediction == pear_five_scaled_df_test_y)
print('\nThe accuracy for log regression liblinear on pearson five attributes is:')
print(round(accuracy * 100, 2), '%')

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

accuracy = np.mean(prediction == pear_ten_scaled_df_test_y)
print('\nThe accuracy for log regression liblinear on pearson ten attributes is:')
print(round(accuracy * 100, 2), '%')

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

accuracy = np.mean(prediction == ols_scaled_df_test_y)
print('\nThe accuracy for log regression liblinear on ols attributes is:')
print(round(accuracy * 100, 2), '%')

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

accuracy = np.mean(prediction == rfe_scaled_df_test_y)
print('\nThe accuracy for log regression liblinear on rfe attributes is:')
print(round(accuracy * 100, 2), '%')

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

accuracy = np.mean(prediction == lasso_scaled_df_test_y)
print('\nThe accuracy for log regression liblinear on lasso attributes is:')
print(round(accuracy * 100, 2), '%')

####
# End lasso dataset
####

#############
# End log regression liblinear solver
#############

#############
# Start log regression sag solver
#############

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

accuracy = np.mean(prediction == pear_five_scaled_df_test_y)
print('\nThe accuracy for log regression sag on pearson five attributes is:')
print(round(accuracy * 100, 2), '%')

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

accuracy = np.mean(prediction == pear_ten_scaled_df_test_y)
print('\nThe accuracy for log regression sag on pearson ten attributes is:')
print(round(accuracy * 100, 2), '%')

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

accuracy = np.mean(prediction == ols_scaled_df_test_y)
print('\nThe accuracy for log regression sag on ols attributes is:')
print(round(accuracy * 100, 2), '%')

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

accuracy = np.mean(prediction == rfe_scaled_df_test_y)
print('\nThe accuracy for log regression sag on rfe attributes is:')
print(round(accuracy * 100, 2), '%')

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

accuracy = np.mean(prediction == lasso_scaled_df_test_y)
print('\nThe accuracy for log regression sag on lasso attributes is:')
print(round(accuracy * 100, 2), '%')

####
# End lasso dataset
####

#############
# End log regression sag solver
#############

#############
# Start log regression newton-cg solver
#############

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

accuracy = np.mean(prediction == pear_five_scaled_df_test_y)
print('\nThe accuracy for log regression newton-cg on pearson five attributes is:')
print(round(accuracy * 100, 2), '%')

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

accuracy = np.mean(prediction == pear_ten_scaled_df_test_y)
print('\nThe accuracy for log regression newton-cg on pearson ten attributes is:')
print(round(accuracy * 100, 2), '%')

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

accuracy = np.mean(prediction == ols_scaled_df_test_y)
print('\nThe accuracy for log regression newton-cg on ols attributes is:')
print(round(accuracy * 100, 2), '%')

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

accuracy = np.mean(prediction == rfe_scaled_df_test_y)
print('\nThe accuracy for log regression newton-cg on rfe attributes is:')
print(round(accuracy * 100, 2), '%')

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

accuracy = np.mean(prediction == lasso_scaled_df_test_y)
print('\nThe accuracy for log regression newton-cg on lasso attributes is:')
print(round(accuracy * 100, 2), '%')

####
# End lasso dataset
####

#############
# End log regression newton-cg solver
#############

################
# End building scaled algorithms
################
