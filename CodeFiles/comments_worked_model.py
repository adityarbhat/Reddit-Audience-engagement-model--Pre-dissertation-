#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:19:06 2019

@author: adityabhat
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:02:17 2019

@author: adityabhat
"""

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
data_comm = pd.read_csv("/Users/adityabhat/Downloads/Datasets/CommentsDatatsets/comments_transformed copy.csv")
features_comm = data_comm.iloc[:,0:30].columns.tolist()

dataset_comm = data_comm.values
# split data into X and y
X = dataset_comm[:,0:30]
Y = dataset_comm[:,30]


seed=7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=test_size, random_state=seed)
# fit model on training data
model_comm = XGBClassifier(max_depth=3, learning_rate=0.08, n_estimators=100,booster='gbtree',importance_type="gain",num_class=3,objective='multi:softmax')
model_comm.fit(X_train, y_train)

# make predictions for test data
predictions_comm = model_comm.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, predictions_comm)
print("Accuracy: %.2f%%" % (accuracy * 100.0))




plot_importance(model_comm,importance_type="gain", color='red',max_num_features=10)
plt.show()
#multiclass auc and roc
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
     lb = MultiLabelBinarizer()
     lb.fit(y_test)
     y_test = lb.transform(y_test)
     y_pred = lb.transform(y_pred)
     return roc_auc_score(y_test, y_pred, average=average)
 
multiclass_roc_auc_score(y_test,predictions_comm)

#classification report
le_c = LabelEncoder()
y = le_c.fit_transform(data_comm.iloc[:, 30])
print(classification_report(y_test, predictions_comm, target_names=le_c.classes_))