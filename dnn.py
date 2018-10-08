
import numpy as np
import tensorflow as tf
import pandas as pd
from keras import initializers, optimizers, metrics
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn import preprocessing
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score
from imblearn.over_sampling import SMOTE

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
    return train.as_matrix(), validate.as_matrix(), test.as_matrix()    # return numpy arrays


'''
DNN Predictor for all the products
Input: Condensed history with the last months
Output: Predicted Product ownership probability over all the products
'''
def dnn_all(file):
    # ======================== Data Information ===========================
    n_features = 184
    n_products = 24
    n_fields = n_features + n_products

    # ======================== Splitting into Sets ========================
    rf = pd.read_csv(file)
    print("%s file read." %file)
    print("Splitting into Train, Validation and Test")
    train, valid, test = train_validate_test_split(rf)

    x_train = train[:, 0:n_features]                # the condensed history     (n_cust, n_features)
    y_train = train[:, n_features:n_fields]         # products                  (n_cust, n_products)

    x_val = valid[:, 0:n_features]
    y_val = valid[:, n_features:n_fields]

    x_test = test[:, 0:n_features]
    y_test = test[:, n_features:n_fields]

    print("Data formatted.")

    # ==================== DNN Model for Single Product ====================
    batch_size = 64
    epochs = 5
    print("Building DNN Predictor Model")

    model = Sequential()
    model.add(Dense(64, input_shape =(n_features,), activation = 'relu', kernel_initializer = 'glorot_normal', bias_initializer = 'random_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(48, activation = 'relu', kernel_initializer = 'glorot_normal', bias_initializer = 'random_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(24, activation = 'relu', kernel_initializer = 'glorot_normal', bias_initializer = 'random_uniform'))
    model.add(Dense(n_products, activation = 'softmax'))

    model.compile(loss ='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'] )

    print("Start Training")

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

    print("Finish Training")

    print("Evaluation Result:")

    scores = model.evaluate(x_test, y_test, batch_size = batch_size)
    print("\n%s: %.5f%%" % (model.metrics_names[1], scores[1]*100))

    y_pred = model.predict(x_test, batch_size = batch_size)

    for i in range(n_products):
        f1 = f1_score(y_test[i], y_pred[i])
        roc = roc_auc_score(y_test[i], y_pred[i])
        kappa = cohen_kappa_score(y_test[i], y_pred[i])
        print ("Product %d : F1_score = %.5f%%  ROC_AUC = %.5f%%  Cohen_Kappa = %.5f%%" %(i, f1, roc, kappa))

'''
DNN Predictor for a single target product
Input: Condensed history with the last months
Output: Predicted Product ownership probability for the target product
'''
def dnn_single(file):

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
    label = "label_" + str(p_idx)                         # Set target product
    print(label)

    x_train = train[:, 0:n_features]                # the condensed history    (n_cust, n_features)
    y_train = train[:, n_features + p_idx]          # the month to predict     (n_cust, 1)

    
    # ======================== SMOTE Oversampling ========================
    y_train = np.ravel(y_train)
    print("Original Dataset: ", binary_counter(y_train))    # count of +ve and -ve labels
    sm = SMOTE(random_state = 1024)
    x_train, y_train = sm.fit_sample(x_train, y_train)
    print("SMOTE Resampled Dataset: ", binary_counter(y_train)) 
    y_train = np.reshape(y_train, (-1,1))


    x_val = valid[:, 0:n_features]
    y_val = valid[:, n_features + p_idx]

    x_test = test[:, 0:n_features]
    y_test = test[:, n_features + p_idx]

    print("Data formatted.")

    # ==================== DNN Model for Single Product ====================
    batch_size = 64
    epochs = 5

    print("Building DNN Single Predictor Model")

    model = Sequential()
    model.add(Dense(64, input_shape =(n_features,), activation = 'relu', kernel_initializer = 'glorot_normal', bias_initializer = 'random_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(48, activation = 'relu', kernel_initializer = 'glorot_normal', bias_initializer = 'random_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(24, activation = 'relu', kernel_initializer = 'glorot_normal', bias_initializer = 'random_uniform'))
    model.add(Dense(1, activation = 'softmax'))

    model.compile(loss ='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'] )

    print("Start Training")

    # With Mortgage Class Weight
    class_weight = {0: 0.5041410854829876, 1: 60.87064461167116}
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, class_weight = class_weight, validation_data=(x_val, y_val))
    
    '''
    # Without Class Weight
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))
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
    y_pred = model.predict(x_test, batch_size = batch_size)

    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    print ("F1_Score = %.5f%%  ROC_AUC = %.5f%%  Cohen_Kappa = %.5f%%" %(f1, roc, kappa))


dnn_all('data2.csv')
#dnn_single('data2.csv')
