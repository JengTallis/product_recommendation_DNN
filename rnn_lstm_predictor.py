# rnn_lstm_predictor.py

import numpy as np
import tensorflow as tf
import pandas as pd
from keras import initializers, optimizers, metrics
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, Activation
from sklearn import preprocessing
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score


def to_binary(xs):
	ys = []
	for x in xs:
		y = 1 if x >= 0.5 else 0
		ys.append(y)
	return ys

def generate_chunk(reader, chunksize):
	chunk = []
	for index, line in enumerate(reader):
		if (index % chunksize == 0 and index > 0):
			yield chunk
			del chunk[:]
		chunk.append(line)
	yield chunk

'''
Take pandas dataframe, 
split into train, validation and test set, 
return the sets as numpy.ndarray
'''
def train_validate_test_split(df, train_percent=.6, val_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    l = len(df)
    train_end = int(train_percent * l)
    val_end = int(val_percent * l) + train_end
    train = df.ix[perm[:train_end]]
    validate = df.ix[perm[train_end:val_end]]
    test = df.ix[perm[val_end:]]
    #return train, validate, test
    return train.as_matrix(), validate.as_matrix(), test.as_matrix()	# return numpy arrays

'''
convert an array of values into a dataset matrix
@param look_back = the number of time steps to look backwards
@return numpy arrays X and Y
'''
'''
def create_dataset(dataset, look_back=12):
	X, Y = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0:40]
		X.append(a)
		Y.append(dataset[i + look_back, 16:40])
	return numpy.array(X), numpy.array(Y)
'''

'''
Format the Data
'''
def rnn_data(file):

	rf = pd.read_csv(file)
	print("%s file read." %file)
	rf.drop(['FetchDate','CusID'], axis=1, inplace=True)

	n_cust_infos = 16
	n_products = 24
	n_fields = n_cust_infos + n_products
	n_months = 17
	timesteps = 16
	start_idx = 0

	data_arr = []

	for chunk in generate_chunk(rf.values, n_months):	# one customer
		flat = np.reshape(chunk,(1,-1))	# one cust as (1, n_fields)
		data_arr.append(flat)

	data = np.array(data_arr)
	#print(data.shape)	# (n_cust, 1, n_months*n_fields)
	data = np.reshape(data,(-1, n_months*n_fields))
	#print(data.shape)	# (n_cust, n_months*n_fields)

	# normalize data
	scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
	data = scaler.fit_transform(data)
	print("Data normalized")

	df = pd.DataFrame(data=data)
	#print(df.shape)		# (n_cust, n_months*n_fields)

	print("Splitting into train, validation, test")
	train, validate, test = train_validate_test_split(df)

	train = np.reshape(train,(-1, n_months, n_fields))
	valid = np.reshape(validate,(-1, n_months, n_fields))
	test = np.reshape(test,(-1, n_months, n_fields))

	x_train = train[:, 0:timesteps, start_idx:n_fields]		# the history months    (n_cust, timesteps, n_fields)
	y_train = train[:, timesteps, n_cust_infos:n_fields]	# the month to predict  (n_cust, n_products)

	x_valid = valid[:, 0:timesteps, start_idx:n_fields]
	y_valid = valid[:, timesteps, n_cust_infos:n_fields]

	x_test = test[:, 0:timesteps, start_idx:n_fields]
	y_test = test[:, timesteps, n_cust_infos:n_fields]

	print("Data formatted.")

	return x_train, y_train, x_valid, y_valid, x_test, y_test

'''
RNN LSTM
'''
def lstm_rnn_predictor(x_train, y_train, x_val, y_val, x_test, y_test):

	# ===================== Data ======================
	np.random.seed(18657865)
	print("Getting Data")

	# ===================== LSTM RNN Model ====================

	batch_size = 64
	epochs = 5

	data_dim = 40	# n_fields = 40
	timesteps = 16	# 16 months
	label_dim = 24	# 24 products

	print("Building RNN LSTM Predictor Model")
	# expected input data shape: (batch_size, timesteps, data_dim)
	model = Sequential()
	model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
	model.add(LSTM(32, return_sequences=False))  # returns only from the last time step
	model.add(Dense(label_dim, activation='softmax'))

	model.compile(loss='mean_squared_error',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	print("Start training")
	model.fit(x_train, y_train,
	          batch_size=batch_size, epochs=epochs,
	          validation_data=(x_val, y_val))
	print("Finish training")

	# ===================== Evaluation ====================
	evaluate_model(model, x_test , y_test, batch_size, label_dim)

'''
BRNN LSTM
'''
def lstm_brnn_predictor(x_train, y_train, x_val, y_val, x_test, y_test):

	# ===================== Data ======================
	np.random.seed(18657865)
	print("Getting Data")

	# ===================== LSTM RNN Model ====================

	batch_size = 64
	epochs = 5

	data_dim = 40	# n_fields = 40
	timesteps = 16	# 16 months
	label_dim = 24	# 24 products

	print("Building BRNN LSTM Predictor Model")
	# expected input data shape: (batch_size, timesteps, data_dim)
	model = Sequential()
	model.add(Bidirectional(LSTM(32, return_sequences=True, kernel_initializer='random_normal'), input_shape=(timesteps, data_dim)))	# returns a sequence of vectors of dimension 32
	model.add(LSTM(32, activation='relu', return_sequences=False))  # returns only from the last time step
	model.add(Dense(label_dim, activation='softmax'))

	'''
	model.compile(loss='mean_squared_error',
	              optimizer='adam',
	              metrics=['accuracy'])
	'''

	model.compile(loss='categorical_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	print("Start training")
	model.fit(x_train, y_train,
	          batch_size=batch_size, epochs=epochs,
	          validation_data=(x_val, y_val))
	print("Finish training")

	# ===================== Evaluation ====================
	evaluate_model(model, x_test , y_test, batch_size, label_dim)

'''
Model Evaluation
Accuracy, F1_score and ROC
'''
def evaluate_model(model, x_test , y_test, batch_size, label_dim):

	print("Evaluation Result:")

	scores = model.evaluate(x_test, y_test, batch_size = batch_size)
	print("\n%s: %.5f%%" % (model.metrics_names[1], scores[1]*100))

	# F1 score (Harmonic mean of precision and recall)
	# ROC
	y_pred = model.predict(x_test, batch_size = batch_size)

	for i in range(label_dim):
		y_pred_b = to_binary(y_pred[i])
		#f1 = f1_score(y_test[i], y_pred_b)
		f1 = -1
		roc = roc_auc_score(y_test[i], y_pred_b)
		kappa = cohen_kappa_score(y_test[i], y_pred_b)
		print ("Product %d : F1_score = %.5f%%  ROC_AUC = %.5f%%  Cohen_Kappa = %.5f%%" %(i, f1, roc, kappa))

	for i in range(label_dim):
		y_pred_b = to_binary(y_pred[i])
		f1 = f1_score(y_test[i], y_pred_b)
		roc = roc_auc_score(y_test[i], y_pred_b)
		kappa = cohen_kappa_score(y_test[i], y_pred_b)
		print ("Product %d : F1_score = %.5f%%  ROC_AUC = %.5f%%  Cohen_Kappa = %.5f%%" %(i, f1, roc, kappa))

x_train, y_train, x_val, y_val, x_test, y_test = rnn_data('senior.csv')	# Segment Data into Train, Validation, Test
lstm_rnn_predictor(x_train, y_train, x_val, y_val, x_test, y_test)
#lstm_brnn_predictor(x_train, y_train, x_val, y_val, x_test, y_test)

