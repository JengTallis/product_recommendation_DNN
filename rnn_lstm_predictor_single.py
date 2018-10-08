# rnn_lstm_predictor_single.py

'''

An LSTM RNN (URNN and BRNN) predictor for a single product

Input: monthly sequence of [customer info + product ownership]
Output: probability of a product being owned and not own for the coming month

Combatting Class Imbalalnce with

1. Cost Function Based Approach : 
	weight[i] = n_samples / (n_classes * bc[i])
	Asymmetric Cost function, cost-sensitive training, penalized model

2. Sampling Based Approach (Resampling) : 
	1) Oversampling: RandomOverSampler, SMOTE, ADASYN
	2) Hybrid(Oversampling + Undersampling): SMOTEEN, SMOTETomek 

'''

import numpy as np
import tensorflow as tf
import pandas as pd
from keras import initializers, optimizers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, Activation
from sklearn import preprocessing
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score	# Skewed Dataset Metrics
from keras.utils.np_utils import to_categorical
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN	# Oversampling
from imblearn.combine import SMOTEENN, SMOTETomek	# Hybrid resampling
# from collections import Counter     				# ndarray not hashable, cannot be used


'''
Convert a probability array to a binary array
Using threshold = 0.5
'''
def to_binary(xs):
	ys = []
	for x in xs:
		y = 1 if x[1] >= 0.5 else 0
		ys.append(y)
	return ys

'''
Count occurrences of binary class
@return counter = {class: count}
'''
def binary_counter(arr):
	bc = [0,0]
	n_samples = 0
	n_classes = 2
	for a in arr:
		n_samples += 1
		bc[int(a)] += 1
	counter = {0 : bc[0], 1: bc[1]}
	return counter

'''
Chunk generator
@param chunksize = the size of chunk to generate
'''
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
Preprocess the Data
Handle Class Imbalance
'''
def rnn_data(file):

	# ======================= Data Information ========================
	rf = pd.read_csv(file)
	print("%s file read." %file)
	rf.drop(['FetchDate','CusID'], axis=1, inplace=True)

	n_cust_infos = 16
	n_products = 24
	n_fields = n_cust_infos + n_products
	n_months = 17
	timesteps = 16
	start_idx = 0

	# ======================= Product List =========================
	products = ["SavingAcnt", "Guarantees",
                "CurrentAcnt", "DerivativeAcnt", "PayrollAcnt", "JuniorAcnt", "MoreParticularAcnt",
                "ParticularAcnt", "ParticularPlusAcnt", "ShortDeposit", "MediumDeposit", "LongDeposit",
                "eAcnt", "Funds", "Mortgage", "Pensions", "Loans",
                "Taxes", "CreditCard", "Securities", "HomeAcnt", "Payroll",
                "PayrollPensions", "DirectDebit"]

	target_product = "MediumDeposit"		# Set target product

	p = products.index(target_product)
	print("Target Product : %s" %target_product)
	product = rf[target_product].as_matrix()

	# ================== Calculate Class Weight ===================
	bc = [0,0]		
	n_samples = 0
	n_classes = 2
	for prd in product:
		n_samples += 1
		bc[int(prd)] += 1		# Count Binary Class
	print("Class Counted")
	print(bc)
	print("Imbalance Ratio : %f" %(float(bc[0])/float(bc[1])))

	weight = [0,0]
	for i in range(n_classes):
		weight[i] = n_samples / (n_classes * bc[i])		# Calcuate Class Weight
	class_weight = {0 : weight[0], 1: weight[1]}
	print("Class Weight Calcuated")
	print(class_weight)

	# ================== Formatting and Normalizing Data ==================
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
	print("Data Normalized")

	# ====================== Splitting into sets =========================

	df = pd.DataFrame(data=data)
	#print(df.shape)		# (n_cust, n_months*n_fields)
	print("Splitting into Train, Validation, Test")
	train, validate, test = train_validate_test_split(df)

	train = np.reshape(train,(-1, n_months, n_fields))
	valid = np.reshape(validate,(-1, n_months, n_fields))
	test = np.reshape(test,(-1, n_months, n_fields))

	# ====================== Resampling Training Set =======================

	x_train = train[:, 0:timesteps, start_idx:n_fields]		# the history months    (n_cust, timesteps, n_fields)
	y_train = train[:, timesteps, n_cust_infos+p]			# the month to predict  (n_cust, n_products)

	x_train = np.reshape(x_train, (-1, timesteps*n_fields))
	y_train = np.reshape(y_train,(-1,1))
	y_train = np.ravel(y_train)

	print("Original Dataset: ", binary_counter(y_train))	# count of +ve and -ve labels

	'''
	# RandomOverSampler Oversampling
	ros = RandomOverSampler(random_state = 1024)
	x_train, y_train = ros.fit_sample(x_train, y_train)
	print("RandomOverSampler Resampled Dataset: ", binary_counter(y_train))	
	'''

	# SMOTE Oversampling
	sm = SMOTE(random_state = 1024)
	x_train, y_train = sm.fit_sample(x_train, y_train)
	print("SMOTE Resampled Dataset: ", binary_counter(y_train))	

	'''
	# ADASYN Oversampling
	ada = ADASYN(random_state = 1024)
	x_train, y_train = ada.fit_sample(x_train, y_train)	# resampling
	print("ADASYN Resampled dataset: ", binary_counter(y_train))

	# SMOTEENN Hybrid
	smote_enn = SMOTEENN(random_state = 1024)
	x_train, y_train = smote_enn.fit_sample(x_train, y_train)
	print("SMOTEENN Resampled Dataset: ", binary_counter(y_train))	

	# SMOTETomek Hybrid
	smo_tomek = SMOTETomek(random_state = 1024)
	x_train, y_train = smo_tomek.fit_sample(x_train, y_train)
	print("SMOTETomek Resampled Dataset: ", binary_counter(y_train))
	'''

	x_train = np.reshape(x_train, (-1, n_months-1, n_fields))
	y_train = np.reshape(y_train, (-1,1))


	# ============ Preparing Validation and Test Set =================

	x_valid = valid[:, 0:timesteps, start_idx:n_fields]
	y_valid = valid[:, timesteps, n_cust_infos+p]

	x_test = test[:, 0:timesteps, start_idx:n_fields]
	y_test = test[:, timesteps, n_cust_infos+p]

	y_train = to_categorical(y_train)
	y_valid = to_categorical(y_valid)
	y_test = to_categorical(y_test)

	print("Data Formatted")

	return x_train, y_train, x_valid, y_valid, x_test, y_test, class_weight

'''
RNN LSTM
'''
def lstm_rnn_predictor(x_train, y_train, x_val, y_val, x_test, y_test, class_weight):

	# ===================== Data ======================
	np.random.seed(18657865)
	print("Getting Data for RNN")

	# ===================== LSTM RNN Model ====================

	batch_size = 64
	epochs = 5

	data_dim = 40	# n_fields = 40
	timesteps = 16	# 16 months
	label_dim = 2	# 1 product, 2 classes

	print("Building RNN LSTM Single Predictor Model")
	# expected input data shape: (batch_size, timesteps, data_dim)
	model = Sequential()
	model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
	model.add(LSTM(32, return_sequences=False))  # returns only from the last time step
	model.add(Dense(label_dim, activation='softmax'))

	model.compile(loss='mean_squared_error',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	print("Start Training")
	model.fit(x_train, y_train,
	          batch_size=batch_size, epochs=epochs, class_weight = class_weight,
	          validation_data=(x_val, y_val))
	print("Finish Training")

	# ===================== Evaluation ====================
	evaluate_model(model, x_test , y_test, batch_size)

'''
BRNN LSTM
'''
def lstm_brnn_predictor(x_train, y_train, x_val, y_val, x_test, y_test, class_weight):

	# ======================== Data ===========================
	np.random.seed(18657865)
	print("Getting Data for RNN")

	# ===================== LSTM RNN Model ====================

	batch_size = 64
	epochs = 5

	data_dim = 40	# n_fields = 40
	timesteps = 16	# 16 months
	label_dim = 2	# 1 product, 2 classes

	print("Building BRNN LSTM Single Predictor Model")
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

	print("Start Training")
	model.fit(x_train, y_train,
	          batch_size=batch_size, epochs=epochs,
	          validation_data=(x_val, y_val))
	'''
	# with class weight
	model.fit(x_train, y_train,
	          batch_size=batch_size, epochs=epochs, class_weight = class_weight,
	          validation_data=(x_val, y_val))
	'''
	print("Finish Training")

	# ===================== Evaluation ====================
	evaluate_model(model, x_test , y_test, batch_size)

'''
Model Evaluation
Accuracy, F1_score and ROC
'''
def evaluate_model(model, x_test , y_test, batch_size):

	print("Evaluation Result:")

	scores = model.evaluate(x_test, y_test, batch_size = batch_size)
	print("\n%s: %.5f%%" % (model.metrics_names[1], scores[1]*100))

	# F1 score (Harmonic mean of precision and recall)
	# ROC
	y_p = model.predict(x_test, batch_size = batch_size)
	#print(y_p.shape)
	#print(y_p)
	y_pred = to_binary(y_p)
	y_tb = to_binary(y_test)

	f1 = f1_score(y_tb, y_pred)
	roc = roc_auc_score(y_tb, y_pred)
	kappa = cohen_kappa_score(y_tb, y_pred)
	print ("F1_Score = %.5f%%  ROC_AUC = %.5f%%  Cohen_Kappa = %.5f%%" %(f1, roc, kappa))


x_train, y_train, x_val, y_val, x_test, y_test, class_weight = rnn_data('senior.csv')	# Segment Data into Train, Validation, Test
#lstm_rnn_predictor(x_train, y_train, x_val, y_val, x_test, y_test, class_weight)
lstm_brnn_predictor(x_train, y_train, x_val, y_val, x_test, y_test, class_weight)

