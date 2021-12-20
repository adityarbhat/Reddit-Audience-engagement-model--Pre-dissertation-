#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:15:54 2019

@author: adityabhat
"""


# binary classification, breast cancer dataset, label and one hot encoded


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot as plt
#from sklearn.metrics import mean_squared_error
# load data
data = pd.read_csv("/Users/adityabhat/Downloads/Datasets/ScoresRelatedDatasets/Scores_Transformed.csv")
'''getting the string varibales in the dataset'''
#obj_df = data.select_dtypes(include=['object']).copy()
#obj_df.head()

#converting the string variables to dummies 
#onehot=pd.get_dummies(obj_df)
#onehot.to_csv("/Users/adityabhat/Downloads/enconding.csv")

#reading the updated csv file

#data_modified=pd.read_csv("/Users/adityabhat/Downloads/Datasets/Upvotes_Dataset.csv")
dataset = data.values
# split data into X and y
X = dataset[:,0:93]
Y = dataset[:,93]
# encode string input values as integers
seed=7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=test_size, random_state=seed)
# fit model on training data
model = XGBClassifier(max_depth=2, learning_rate=0.08, n_estimators=100,booster='gbtree',importance_type="gain")
model.fit(X_train, y_train)
#print(model)
# make predictions for test data
predictions = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#rmse = np.sqrt(mean_squared_error(y_test, predictions))
#pyplot.bar(range(len(model.feature_importances_)),model.feature_importances_)
#pyplot.show()
#print("RMSE: %f" % (rmse))
features = data.iloc[:,0:93].columns.tolist()
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
#names = [data.feature_names[i] for i in indices]
plt.bar(range(X.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(X.shape[1]), rotation=20, fontsize = 8)
# Create plot title
plt.title("Feature Importance")
# Show plot
plt.show()

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)),features.columns[indices])
plt.xlabel('Relative Importance')
plt.show()

plot_importance(model,importance_type="gain", color='red')
plt.show()

