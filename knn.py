'''
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
knn.py

KNN imputation on the dataset
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

import numpy as np
from sklearn import preprocessing, neighbors, model_selection
import pandas as pd

rf = pd.read_csv('knn_train.csv')
rf.drop(['FetchDate','CusID'], axis=1, inplace=True)
rf.replace('NA', -99999, inplace=True)

# define features and labels
data = rf.values
X = data[:,0:16]
y = data[:,16:40]

X_n = preprocessing.normalize(X, norm='l2', axis=1, copy=True) # nomalization does not help

# Shuffle and split data into training and testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

classifier = neighbors.KNeighborsClassifier()
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
print(accuracy)



