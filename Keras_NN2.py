# Import all required libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.utils import class_weight
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import class_weight

# fix random seed for reproducibility
np.random.seed(7)

#Data loading (Only input features related to product and make predictions for that product)
csvfile = pd.read_csv('output.csv')

data = csvfile[['CusTime', 'EmployeeIdx', 'CntyOfResidence', 'Sex',	'Age', 'NewCusIdx', 'Seniority', 'CusType', 'RelationType',	'ForeignIdx', 'ChnlEnter', 'DeceasedIdx', 'ProvCode',
				'ActivIdx', 'Income', 'Segment', 'PayrollAcnt', '(00)_4', '(01)_4', '(10)_4', '(11)_4', '(0s)_4', '(1s)_4']]

label = csvfile[['label_4']]

#Split datset into training and test data in ratio 7:3
X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size = 0.30, random_state = 7)

#Convert to numpy array 
def preprocess(X):
	return np.array(X, dtype=np.float32)

X_train = preprocess(X_train)
X_test = preprocess(X_test)

#Create neural network model
model = Sequential()
model.add(Dense(64, input_shape =(184,), activation = 'relu', kernel_initializer = 'glorot_normal', bias_initializer = 'random_uniform'))
model.add(Dropout(0.5))
model.add(Dense(48, activation = 'relu', kernel_initializer = 'glorot_normal', bias_initializer = 'random_uniform'))
model.add(Dropout(0.5))
model.add(Dense(24, activation = 'relu', kernel_initializer = 'glorot_normal', bias_initializer = 'random_uniform'))
model.add(Dense(1, activation = 'softmax'))

# Compile model, use Adam Optimizer
model.compile(loss ='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'] )

# Fit the model
model.fit(X_train, Y_train, epochs = 15, batch_size = 100, validation_data = (X_test, Y_test))

# evaluate the model
Kerascores = model.evaluate(X_test, Y_test, batch_size = 100)
print("\n%s: %.5f%%" % (model.metrics_names[1], scores[1]*100))

#F1 score
Y_pred = model.predict(X_test)
f1 = f1_score(Y_test, Y_pred)
print (f1)



