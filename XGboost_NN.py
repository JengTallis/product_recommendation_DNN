# First XGBoost model for Pima Indians dataset
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
# load data
dataset = pd.read_csv('output.csv', header = 0)
# split data into X and y
data = dataset.values
X = data[:,0:184]
Y = data[:,186]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state = seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
#predictions = [round(value) for value in y_pred]
# evaluate predictions
f1 = f1_score(y_test, y_pred)
print(f1)
accuracy = accuracy_score(y_test, y_pred)
print (accuracy)

print (np.mean(y_pred[1:100] == y_test[1:100]))
