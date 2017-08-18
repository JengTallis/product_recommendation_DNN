# rnn_lstm_allmonths.py

import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, TimeDistributed
from sklearn import preprocessing


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

	x_train = train[:, 0:timesteps, n_cust_infos:n_fields]		# the history months    (n_cust, timesteps, n_fields)
	y_train = train[:, timesteps, n_cust_infos:n_fields]	# the month to predict  (n_cust, n_products)

	x_valid = valid[:, 0:timesteps, n_cust_infos:n_fields]
	y_valid = valid[:, timesteps, n_cust_infos:n_fields]

	x_test = test[:, 0:timesteps, n_cust_infos:n_fields]
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

	data_dim = 24	# n_fields = 40
	timesteps = 16	# 16 months
	label_dim = 24	# 24 products

	print("Building RNN LSTM Model")
	# expected input data shape: (batch_size, timesteps, data_dim)
	model = Sequential()
	model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
	model.add(LSTM(32, return_sequences=False))  # returns a sequence of vectors of dimension 32
	model.add(Dense(label_dim, activation='softmax'))

	model.compile(loss='mean_squared_error',
	              optimizer='adam',
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


x_train, y_train, x_val, y_val, x_test, y_test = rnn_data('senior.csv')	# Segment Data into Train, Validation, Test
lstm_rnn_predictor(x_train, y_train, x_val, y_val, x_test, y_test)


