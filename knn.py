'''
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
knn.py

KNN imputation on the dataset
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

import numpy as np
from sklearn import preprocessing, neighbors, model_selection
import pandas as pd
import tensorflow as tf

def knn(data):
	rf = pd.read_csv(data)
	rf.drop(['FetchDate','CusID'], axis=1, inplace=True)
	rf.replace('NA', -99999, inplace=True)

	# define features and labels
	data = rf.values
	X = data[:,0:16]
	y = data[:,16:40]

	X_n = preprocessing.normalize(X, norm='l2', axis=1, copy=True) # nomalization does not help

	# Shuffle and split data into training and testing
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X_n, y, test_size=0.2, random_state=0)

	'''
	classifiers = []
	ks = [3,5]
	for k in ks:
		print("k = %d" %k)
		#classifier = neighbors.KNeighborsClassifier(k)
		#print("KNN classifier created")
		dclr = neighbors.KNeighborsClassifier(k,'distance')
		print("distance KNN classifier created")

		radiusClfr = neighbors.RadiusNeighborsClassifier(k)
		print("radius KNN classifier created")

		#bt = neighbors.KNeighborsClassifier(k, 'uniform', 'ball_tree')
		#print("ball tree classifier created")

		#kdt = neighbors.KNeighborsClassifier(k,'uniform', 'kd_tree')
		#print("kd_tree classifier created")

		#classifiers.append(classifier)
		classifiers.append(dclr)
		classifiers.append(radiusClfr)
		#classifiers.append(bt)
		#classifiers.append(kdt)

	for classifier in classifiers:
		classifier.fit(X_train, y_train)
		accuracy = classifier.score(X_test, y_test)
		print(accuracy)
	'''

	clr = neighbors.KNeighborsClassifier(3,'distance')
	print("Distance KNN classifier created.")
	clr.fit(X_n, y)
	print("Finish training.")
	return clr

def predict(classifier, data):
	fields = ["SavingAcnt", "Guarantees",
                "CurrentAcnt", "DerivativeAcnt", "PayrollAcnt", "JuniorAcnt", "MoreParticularAcnt",
                "ParticularAcnt", "ParticularPlusAcnt", "ShortDeposit", "MediumDeposit", "LongDeposit",
                "eAcnt", "Funds", "Mortgage", "Pensions", "Loans",
                "Taxes", "CreditCard", "Securities", "HomeAcnt", "Payroll",
                "PayrollPensions", "DirectDebit"]

	rf = pd.read_csv(data)
	print("%s file read." %data)

	copy = rf.copy(deep=True)									# copy the customer info
	copy.drop(fields,inplace=True, axis=1)
	print("Copy made.")

	rf.drop(['FetchDate','CusID'], axis=1, inplace=True)
	X = rf.values[:,0:16]
	X_n = preprocessing.normalize(X, norm='l2', axis=1, copy=True)	# normalize features
	print("Finish normalization.")

	os = []
	# for row in X_n.itertuples():
	for row in X_n:
		a = []
		a.append(row)
		o = classifier.predict(a)
		os.append(o[0])
	
	# os = classifier.predict(X)
										# predict
	print("Prediction made.")
	od = pd.DataFrame(data=os,columns=fields)

	result = pd.concat([copy, od], axis=1)		# concatenate the prediction after the customer info
	result.to_csv('imputed.csv', index=False)	# write to csv file

classifier = knn('knn_train.csv')
predict(classifier, "missing.csv")

#predict(classifier, "predict.csv") # for testing purpose





