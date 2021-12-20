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


import pandas as pd

from xgboost import XGBClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report

from xgboost import plot_importance
from matplotlib import pyplot as plt
#from sklearn.metrics import mean_squared_error
# load data
data = pd.read_csv("/Users/adityabhat/Downloads/Datasets/ScoresRelatedDatasets/Scores_TransformedV2 copy.csv")
features = data.iloc[:,0:30].columns.tolist()

#obj_df = data.select_dtypes(include=['object']).copy()
#obj_df.head()

#converting the string variables to dummies 
#onehot=pd.get_dummies(obj_df)
#onehot.to_csv("/Users/adityabhat/Downloads/enconding.csv")

#reading the updated csv file

#data_modified=pd.read_csv("/Users/adityabhat/Downloads/Datasets/Upvotes_Dataset.csv")
dataset = data.values
# split data into X and y
X = dataset[:,0:30]
Y = dataset[:,30]
le = LabelEncoder()
y = le.fit_transform(data.iloc[:, 30])
# encode string input values as integers
seed=7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=test_size, random_state=seed)



# fit model on training data
model = XGBClassifier(max_depth=2, learning_rate=0.08, n_estimators=100,booster='gbtree',importance_type="gain",num_class=3,objective='multi:softmax')
model.fit(X_train, y_train)
#print(model)

predictions = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


#rmse = np.sqrt(mean_squared_error(y_test, predictions))
#pyplot.bar(range(len(model.feature_importances_)),model.feature_importances_)
#pyplot.show()
#print("RMSE: %f" % (rmse))

print(classification_report(y_test, predictions, target_names=le.classes_))

plot_importance(model,importance_type="gain", color='red',max_num_features=10)
plt.show()

#multiclass auc and roc
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
     lb = MultiLabelBinarizer()
     lb.fit(y_test)
     y_test = lb.transform(y_test)
     y_pred = lb.transform(y_pred)
     return roc_auc_score(y_test, y_pred, average=average)
 
multiclass_roc_auc_score(y_test,predictions)