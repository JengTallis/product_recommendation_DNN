# Create your first MLP in Keras
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

csvfile = pd.read_csv('output.csv')

data = csvfile[['CusTime', 'EmployeeIdx', 'CntyOfResidence', 'Sex',	'Age', 'NewCusIdx', 'Seniority', 'CusType', 'RelationType',	'ForeignIdx', 'ChnlEnter', 'DeceasedIdx', 'ProvCode',
				'ActivIdx', 'Income', 'Segment', 'SavingAcnt', 'Guarantees', 'CurrentAcnt', 'DerivativeAcnt', 'PayrollAcnt', 'JuniorAcnt', 'MoreParticularAcnt', 'ParticularAcnt',
				'ParticularPlusAcnt', 'ShortDeposit', 'MediumDeposit', 'LongDeposit', 'eAcnt', 'Funds', 'Mortgage', 'Pensions',	'Loans', 'Taxes', 'CreditCard',	'Securities', 'HomeAcnt',
				'Payroll', 'PayrollPensions', 'DirectDebit', '(00)_0', '(01)_0', '(10)_0', '(11)_0', '(0s)_0', '(1s)_0', '(00)_1', '(01)_1', '(10)_1', '(11)_1', '(0s)_1', '(1s)_1', 
				'(00)_2', '(01)_2', '(10)_2', '(11)_2', '(0s)_2', '(1s)_2', '(00)_3', '(01)_3', '(10)_3', '(11)_3', '(0s)_3', '(1s)_3', '(00)_4', '(01)_4', '(10)_4', '(11)_4', '(0s)_4',
				'(1s)_4', '(00)_5', '(01)_5', '(10)_5', '(11)_5', '(0s)_5', '(1s)_5', '(00)_6', '(01)_6', '(10)_6', '(11)_6', '(0s)_6', '(1s)_6', '(00)_7', '(01)_7', '(10)_7', '(11)_7',
				'(0s)_7', '(1s)_7', '(00)_8', '(01)_8', '(10)_8', '(11)_8', '(0s)_8', '(1s)_8', '(00)_9', '(01)_9', '(10)_9', '(11)_9', '(0s)_9', '(1s)_9', '(00)_10', '(01)_10', 
				'(10)_10', '(11)_10', '(0s)_10', '(1s)_10', '(00)_11', '(01)_11', '(10)_11', '(11)_11', '(0s)_11', '(1s)_11', '(00)_12', '(01)_12', '(10)_12', '(11)_12', '(0s)_12', 
				'(1s)_12', '(00)_13', '(01)_13', '(10)_13', '(11)_13', '(0s)_13', '(1s)_13', '(00)_14', '(01)_14', '(10)_14', '(11)_14', '(0s)_14', '(1s)_14', '(00)_15', '(01)_15', 
				'(10)_15', '(11)_15', '(0s)_15', '(1s)_15', '(00)_16', '(01)_16', '(10)_16', '(11)_16', '(0s)_16', '(1s)_16', '(00)_17', '(01)_17', '(10)_17', '(11)_17', '(0s)_17', 
				'(1s)_17', '(00)_18', '(01)_18', '(10)_18', '(11)_18', '(0s)_18', '(1s)_18', '(00)_19', '(01)_19', '(10)_19', '(11)_19', '(0s)_19', '(1s)_19', '(00)_20', '(01)_20', 
				'(10)_20', '(11)_20', '(0s)_20', '(1s)_20', '(00)_21', '(01)_21', '(10)_21', '(11)_21', '(0s)_21', '(1s)_21', '(00)_22', '(01)_22', '(10)_22', '(11)_22', '(0s)_22', 
				'(1s)_22', '(00)_23', '(01)_23', '(10)_23', '(11)_23', '(0s)_23', '(1s)_23']]
'''
label = csvfile[['label_0', 'label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6', 'label_7', 'label_8', 'label_9', 'label_10', 'label_11', 'label_12',
				'label_13', 'label_14',	'label_15',	'label_16', 'label_17', 'label_18',	'label_19',	'label_20',	'label_21',	'label_22',	'label_23']]

data = csvfile[['CusTime', 'EmployeeIdx', 'CntyOfResidence', 'Sex',	'Age', 'NewCusIdx', 'Seniority', 'CusType', 'RelationType',	'ForeignIdx', 'ChnlEnter', 'DeceasedIdx', 'ProvCode',
				'ActivIdx', 'Income', 'Segment', 'PayrollAcnt', '(00)_4', '(01)_4', '(10)_4', '(11)_4', '(0s)_4', '(1s)_4']]
'''
label = csvfile[['label_0']]

print('Original dataset shape {}'.format(Counter(label)))
sm = SMOTE(random_state = 42)
label = np.ravel(label)
X, Y = sm.fit_sample(data, label)
print('Resampled dataset shape {}'.format(Counter(Y_res))) 

def preprocess(X):
	return np.array(X, dtype=np.float32)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 7)
#class_weight = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)

X_train = preprocess(X_train)
X_test = preprocess(X_test)


# create model
model = Sequential()
model.add(Dense(64, input_shape =(184,), activation = 'relu', kernel_initializer = 'glorot_normal', bias_initializer = 'random_uniform'))
model.add(Dropout(0.5))
model.add(Dense(48, activation = 'relu', kernel_initializer = 'glorot_normal', bias_initializer = 'random_uniform'))
model.add(Dropout(0.5))
model.add(Dense(24, activation = 'relu', kernel_initializer = 'glorot_normal', bias_initializer = 'random_uniform'))
model.add(Dense(1, activation = 'softmax'))

# Compile model
model.compile(loss ='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'] )

# Fit the model
model.fit(X_train, Y_train, epochs = 15, batch_size = 100, validation_data = (X_test, Y_test))

# evaluate the model
#scores = model.evaluate(X_test, Y_test, batch_size = 100)
#print("\n%s: %.5f%%" % (model.metrics_names[1], scores[1]*100))
Y_pred = model.predict(X_test)
f1 = f1_score(Y_test, Y_pred)
print (f1)



