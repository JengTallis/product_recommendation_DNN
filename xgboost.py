'''
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
xgboost.py

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score, accuracy_score
import xgboost as xgb

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
Model Evaluation
Accuracy, F1_score and ROC
'''
def evaluate_model(model, x_test , y_test, batch_size):

	print("Evaluation Result:")

	scores = model.evaluate(x_test, y_test, batch_size = batch_size)
	print("\n%s: %.5f%%" % (model.metrics_names[1], scores[1]*100))

	# F1 score (Harmonic mean of precision and recall)
	# ROC
	y_pred = model.predict(x_test, batch_size = batch_size)


	f1 = f1_score(y_test, y_pred)
	roc = roc_auc_score(y_test, y_pred)
	kappa = cohen_kappa_score(y_test, y_pred)
	print ("F1_Score = %.5f%%  ROC_AUC = %.5f%%  Cohen_Kappa = %.5f%%" %(f1, roc, kappa))


def xg_boost(file):

    # ======================== Data Information ===========================
    n_features = 184
    n_products = 24
    n_fields = n_features + n_products
    products = ["SavingAcnt", "Guarantees",
                "CurrentAcnt", "DerivativeAcnt", "PayrollAcnt", "JuniorAcnt", "MoreParticularAcnt",
                "ParticularAcnt", "ParticularPlusAcnt", "ShortDeposit", "MediumDeposit", "LongDeposit",
                "eAcnt", "Funds", "Mortgage", "Pensions", "Loans",
                "Taxes", "CreditCard", "Securities", "HomeAcnt", "Payroll",
                "PayrollPensions", "DirectDebit"]

    # ======================== Splitting into Sets ========================
    rf = pd.read_csv(file)
    print("%s file read." %file)
    print("Splitting into Train, Validation and Test")
    train, valid, test = train_validate_test_split(rf)

    # ======================== Set Target Product ========================
    target_product = "Mortgage"
    print("Target Product : %s" %target_product, end = " ")
    p_idx = products.index(target_product)
    label = "label" + str(p_idx)                         # Set target product
    print(label)

    x_train = train[:, 0:n_features]                # the condensed history    (n_cust, n_features)
    y_train = train[:, n_features + p_idx]          # the month to predict     (n_cust, 1)

    x_valid = valid[:, 0:n_features]
    y_valid = valid[:, n_features + p_idx]

    x_test = test[:, 0:n_features]
    y_test = test[:, n_features + p_idx]

    print("Data formatted.")

    # ==================== DNN Model for Single Product ====================
    batch_size = 64
    epochs = 5

    print("Building XGboost Single Predictor Model")

    model = xgb.XGBClassifier()

    print("Start Training")

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

    print("Finish Training")

    # ===================== Evaluation ====================
    evaluate_model(model, x_test , y_test, batch_size)

xg_boost('data2.csv')

