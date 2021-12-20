#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:29:47 2019

@author: adityabhat
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import mean_squared_error
# load data
r = pd.read_csv("/Users/adityabhat/Downloads/Datasets/Ranking.csv")
r.columns=r.keys()

'''getting the string varibales in the dataset'''
#obj_df = r.select_dtypes(include=['object']).copy()
#obj_df.head()

#converting the string variables to dummies 
#onehot=pd.get_dummies(obj_df)
#onehot.to_csv("/Users/adityabhat/Downloads/enconding.csv")

#reading the updated csv file

#data_modified=pd.read_csv("/Users/adityabhat/Downloads/Datasets/Ranking.csv")


#dataset = data_modified.values
# split data into X and y
#X = dataset[:,0:165]
#Y = dataset[:,165]
X, Y = r.iloc[:,:-1],r.iloc[:,-1]
data_dmatrix = xgb.DMatrix(data=X,label=Y)

seed=7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=test_size, random_state=seed)
# fit model on training data
model_r = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
model_r.fit(X_train, y_train)

# make predictions for test data
predictions_r = model_r.predict(X_test)
# evaluate predictions
rmse = np.sqrt(mean_squared_error(y_test, predictions_r))
print("RMSE: %.2f%%" % (rmse * 100.0))
var=explained_variance_score(y_test, predictions_r, multioutput='uniform_average')
print("R-squared: %.2f%%" % (var * 100.0))

params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.08,
                'max_depth': 5, 'alpha': 10}
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=7)
cv_results.head()


print((cv_results["test-rmse-mean"]).tail(1))


xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

#plotting xgb trees
xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()

xgb.plot_importance(xg_reg,max_num_features=15)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()


