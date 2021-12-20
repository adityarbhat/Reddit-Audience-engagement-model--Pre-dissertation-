

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:15:54 2019

@author: adityabhat
"""


# binary classification, breast cancer dataset, label and one hot encoded

#from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
#from sklearn.metrics import mean_squared_error
# load data
data_rc = pd.read_csv("/Users/adityabhat/Downloads/Datasets/Ranking/rank_test_main copy.csv")
features_rc = data_rc.iloc[:,0:31].columns.tolist()

dataset_rc = data_rc.values
# split data into X and y
X = dataset_rc[:,0:31]
Y = dataset_rc[:,31]


seed=7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=test_size, random_state=seed)
# fit model on training data
model_rc = XGBClassifier(max_depth=2, learning_rate=0.008, n_estimators=100,booster='gbtree',importance_type="gain")
model_rc.fit(X_train, y_train)

# make predictions for test data
predictions_rc = model_rc.predict(X_test)
# evaluate predictions
accuracy_rc = accuracy_score(y_test, predictions_rc)
print("Accuracy: %.2f%%" % (accuracy_rc * 100.0))




plot_importance(model_rc,importance_type="gain", color='red',max_num_features=10)
plt.show()
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
     lb = MultiLabelBinarizer()
     lb.fit(y_test)
     y_test = lb.transform(y_test)
     y_pred = lb.transform(y_pred)
     return roc_auc_score(y_test, y_pred, average=average)
 
multiclass_roc_auc_score(y_test,predictions_rc)

le_r = LabelEncoder()
y = le_r.fit_transform(data_rc.iloc[:, 31])
print(classification_report(y_test, predictions_rc, target_names=le_r.classes_))

r=roc_auc_score(y_test,predictions_rc)
print("ROC: %.2f%%" % (r*100.0))