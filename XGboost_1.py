#Import all required libraries
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

#Data loading
dataset = pd.read_csv('output.csv', header = 0)

# Splitting data into X and Y
data = dataset.values
X = data[:,0:184]
Y = data[:,184]

# split data into train and test sets in ratio 2:1
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state = seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)

# evaluate predictions
print (np.mean(y_pred == y_test))

#F1 score
f1 = f1_score(y_test, y_pred)
print("f1 score :" , f1)

