# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:12:02 2019
@author: Eric Born

Random saerch for best RF implementation
"""
 Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 5, stop = 250, num = 10)]

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 50, num = 6)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

 Create a RF regressor
rf = RandomForestRegressor(random_state = 1337)

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, 
                               param_distributions = random_grid, 
                               n_iter = 50, cv = 3, verbose=2, 
                               random_state=1337, n_jobs = -1)

#pear_five_df_train_x
#pear_five_df_test_x
#pear_five_df_train_y
#pear_five_df_test_y


# Fit the random search model
rf_random.fit(pear_five_df_train_x, pear_five_df_train_y)

print(rf_random.best_params_)

# store the best estimator
best_random = rf_random.best_estimator_

# calculate prediction percent
pred = 100-(round(np.mean(rf_random.predict(pear_five_df_test_x) 
                                      != pear_five_df_test_y) 
                                      * 100, 2))