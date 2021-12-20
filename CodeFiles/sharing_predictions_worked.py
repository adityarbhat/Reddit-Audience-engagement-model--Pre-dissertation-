
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:15:54 2019

@author: adityabhat
"""


# binary classification, breast cancer dataset, label and one hot encoded

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
data_sharing = pd.read_csv("/Users/adityabhat/Downloads/Datasets/Share_Datasets/sharing_transformed copy.csv")
features_sharing = data_sharing.iloc[:,0:30].columns.tolist()

dataset_sharing = data_sharing.values
# split data into X and y
X = dataset_sharing[:,0:30]
Y = dataset_sharing[:,30]


seed=7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=test_size, random_state=seed)
# fit model on training data
model_s = XGBClassifier(max_depth=2, learning_rate=0.008, n_estimators=100,booster='gbtree',importance_type="gain",num_class=3,objective='multi:softmax')
model_s.fit(X_train, y_train)

# make predictions for test data
predictions_s = model_s.predict(X_test)
# evaluate predictions
accuracy_s = accuracy_score(y_test, predictions_s)
print("Accuracy: %.2f%%" % (accuracy_s * 100.0))




plot_importance(model_s,importance_type="gain", color='red',max_num_features=10)
plt.show()

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
     lb = MultiLabelBinarizer()
     lb.fit(y_test)
     y_test = lb.transform(y_test)
     y_pred = lb.transform(y_pred)
     return roc_auc_score(y_test, y_pred, average=average)
 
le_s = LabelEncoder()
y = le_s.fit_transform(data_sharing.iloc[:, 30])
print(classification_report(y_test, predictions_s, target_names=le_s.classes_))
 
multiclass_roc_auc_score(y_test,predictions_s)