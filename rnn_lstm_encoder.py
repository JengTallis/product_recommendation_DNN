# rnn_lstm_encoder.py

import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, TimeDistributed
from sklearn import preprocessing
from sklearn.metrics import f1_score


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
Format the Data
'''
def rnn_data(file):

	rf = pd.read_csv(file)
	print("%s file read." %file)
	rf.drop(['FetchDate','CusID'], axis=1, inplace=True)

	n_products = 24
	n_months = 17
	timesteps = 12
	start_idx = 0

	data_arr = []
	target_months = []

	for chunk in generate_chunk(rf.values, n_months):	# one customer
		target_month = chunk[n_months-1]	# target month's ground truth
		target_months.append(target_month)
		c = chunk[start_idx:start_idx+timesteps]
		flat = np.reshape(c,(1,-1))	# one cust as (1,480)
		data_arr.append(flat)

	data = np.array(data_arr)
	#print(data.shape)	# (504367, 1, 480)
	data = np.reshape(data,(-1, 480))
	#print(data.shape)	# (504367, 480)

	# normalize data
	scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
	data = scaler.fit_transform(data)
	print("Data normalized")

	df = pd.DataFrame(data=data)
	#print(df.shape)		# (504367, 480)

	print("Splitting into train, validation, test")
	train, validate, test = train_validate_test_split(df)

	train = np.reshape(train,(-1, 12, 40))
	valid = np.reshape(validate,(-1, 12, 40))
	test = np.reshape(test,(-1, 12, 40))

	x_train = train[:,:,0:16]
	y_train = train[:,:,16:40]

	x_valid = valid[:,:,0:16]
	y_valid = valid[:,:,16:40]

	x_test = test[:,:,0:16]
	y_test = test[:,:,16:40]

	print("Data formatted.")

	return x_train, y_train, x_valid, y_valid, x_test, y_test

'''
RNN LSTM
'''
def lstm_rnn(x_train, y_train, x_val, y_val, x_test, y_test):

	# ===================== Data ======================
	np.random.seed(18657865)
	print("Getting Data")

	'''
	# Generate dummy training data
	x_train = np.random.random((1000, timesteps, data_dim))
	y_train = np.random.random((1000, timesteps, label_dim))

	# Generate dummy test data
	x_test = np.random.random((200, timesteps, data_dim))
	y_test = np.random.random((200, timesteps, label_dim))

	# Generate dummy validation data
	x_val = np.random.random((100, timesteps, data_dim))
	y_val = np.random.random((100, timesteps, label_dim))
	'''

	# ===================== LSTM RNN Model ====================

	batch_size = 64
	epochs = 5

	data_dim = 16	# 16 cust info
	timesteps = 12	# 12 months
	label_dim = 24	# 24 products

	print("Building RNN LSTM Encoder Model")
	# expected input data shape: (batch_size, timesteps, data_dim)
	model = Sequential()
	model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
	model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
	model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
	model.add(LSTM(label_dim, return_sequences=True, activation='softmax'))	# returns a softmax sequence of vectors of dimension label_dim 

	model.compile(loss='categorical_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	print("Start training")
	model.fit(x_train, y_train,
	          batch_size=batch_size, epochs=epochs,
	          validation_data=(x_val, y_val))
	print("Finish training")

	# ===================== Evaluation ====================

	print("Evaluattion Result:")
	# evaluate the model
	scores = model.evaluate(x_test, y_test, batch_size = batch_size)
	print("\n%s: %.5f%%" % (model.metrics_names[1], scores[1]*100))

'''
Bidirectional RNN LSTM
'''
def lstm_brnn(x_train, y_train, x_val, y_val, x_test, y_test):

	# ===================== Data ======================
	np.random.seed(18657865)
	print("Getting Data")

	# ===================== LSTM BRNN Model ====================

	batch_size = 64
	epochs = 5

	data_dim = 16	# 16 cust info
	timesteps = 12	# 12 months
	label_dim = 24	# 24 products

	print("Building BRNN LSTM Encoder Model")
	# expected input data shape: (batch_size, timesteps, data_dim)
	model = Sequential()
	model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
	model.add(TimeDistributed(Dense(label_dim, activation='sigmoid')))
	#model.add(LSTM(label_dim, return_sequences=True, activation='softmax'))	# returns a softmax sequence of vectors of dimension label_dim 

	model.compile(loss='categorical_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	print("Start training")
	model.fit(x_train, y_train,
	          batch_size=batch_size, epochs=epochs,
	          validation_data=(x_val, y_val))
	print("Finish training")

	# ===================== Evaluation ====================

	print("Evaluation Result:")
	# evaluate the model
	scores = model.evaluate(x_test, y_test, batch_size = batch_size)
	print("\n%s: %.5f%%" % (model.metrics_names[1], scores[1]*100))

	#F1 score (Harmonic mean of precision and recall)
	y_pred = model.predict(x_test, batch_size = batch_size)

	preds = []
	truth = []
	for i in range(label_dim):
		preds.append(y_pred[i,:])
		truth.append(y_test[i,:])
	print("prediction shape:")
	print(np.shape(preds))
	print(np.shape(y_test))

	for i in range(label_dim):
		f1 = f1_score(truth[i], preds[i])
		print ("F1 score for the %d product: %.5f%%" %(i, f1))

x_train, y_train, x_valid, y_valid, x_test, y_test = rnn_data('senior.csv')	# Segment Data into Train, Validation, Test
#lstm_rnn(x_train, y_train, x_valid, y_valid, x_test, y_test)	# RNN LSTM
lstm_brnn(x_train, y_train, x_valid, y_valid, x_test, y_test)	# BRNN LSTM


