# Import all required libraries
import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import f1_score


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#Data loading 
csvfile = pd.read_csv('output.csv')

dataset = csvfile.values
row_count = dataset.shape[0]

X = dataset[:, 0:184]        #Input features
Y = dataset[:, 184]          #First product

#Using SMOTE
print('Original dataset shape {}'.format(Counter(Y)))          
sm = SMOTE(random_state = 42)
Y = np.ravel(Y)
X_res, Y_res = sm.fit_sample(X, Y)
print('Resampled dataset shape {}'.format(Counter(Y_res)))


# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(64, input_dim = 184, kernel_initializer = 'glorot_normal', activation='relu', bias_initializer = 'random_uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(48, activation = 'relu', kernel_initializer = 'glorot_normal', bias_initializer = 'random_uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(24, activation = 'relu', kernel_initializer = 'glorot_normal', bias_initializer = 'random_uniform'))
	model.add(Dense(1, kernel_initializer = 'glorot_normal', activation ='softmax'))
	# Compile model
	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	return model


# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))              #Data Preprocessing that makes mean 0 and std 1
estimators.append(('mlp', KerasClassifier(build_fn = create_baseline, epochs = 15, batch_size = 100, verbose = 0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)   #Perform K-fold stratified cross validation (K = 10 in this case)
results = cross_val_score(pipeline, X, Y, cv = kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100)) 




