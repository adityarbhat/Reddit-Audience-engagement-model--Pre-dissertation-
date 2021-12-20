#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:24:31 2019

@author: adityabhat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
import math
from __future__ import division
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn import  tree, linear_model
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import explained_variance_score


data = pd.read_csv('/Users/adityabhat/Downloads/scores_Transformedv1.csv')

# Check the number of data points in the data set
print(len(data))
# Check the number of features in the data set
print(len(data.columns))
# Check the data types
print(data.dtypes.unique())

features = data.iloc[:,0:93].columns.tolist()
target = data.iloc[:,93].name

correlations = {}
for f in features:
    data_temp = data[[f,target]]
    x1 = data_temp[f].values
    x2 = data_temp[target].values
    key = f + ' vs ' + target
    correlations[key] = pearsonr(x1,x2)[0]

data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index]


regr = linear_model.LinearRegression()
new_data = data[['comms_num','cross_posts','Life_Span_Short','Life_Span_Medium','Type_Of_Day_Weekday','Type_Of_Day_Weekend',"active_users",'Life_Span_Long','TimeOfDay_Morning','TimeOfDay_Night']]
X = new_data.values
y = data.diff_scores.values

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2)
regr.fit(X_train, y_train)
print(regr.predict(X_test))
regr.score(X_test,y_test)


xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=2)
traindf, testdf = train_test_split(X_train, test_size = 0.3)
xgb.fit(X_train,y_train)

predictions = xgb.predict(X_test)
print(explained_variance_score(predictions,y_test))
