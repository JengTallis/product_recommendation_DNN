keras_rnn.py

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection

import keras.layers as L
import keras.models as M

import numpy

numpy.random.seed(7)

file = ''

rf = pd.read_csv(file)
rf.drop(['CusID'], axis=1, inplace = true)
dataset = rf.values
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# RNN model
model = Sequential()
#model.add(LSTM(24, input_shape=(12, 17)))
model.add(L.LSTM(24, activation='relu', input_dim=17, use_bias=True, input_length=12, return_sequences=True))
#keras.layers.recurrent.LSTM(24, activation='relu', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
model.add(L.LSTM(48, activation = 'relu', return_sequences=True)
model.add(L.LSTM(24, activation = 'relu',  return_sequences=True)
model.add(Dense(1, activation = 'softmax'))


model.compile(loss ='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'] )
model.fit(X_train, Y_train, epochs = 15, batch_size = 100, validation_data = (X_test, Y_test))

# evaluate the model
scores = model.evaluate(X_test, Y_test, batch_size = 100)
print("\n%s: %.5f%%" % (model.metrics_names[1], scores[1]*100))

#F1 score (Harmonic mean of precision and recall)
Y_pred = model.predict(X_test)
f1 = f1_score(Y_test, Y_pred)
print (f1)




# The inputs to the model.
# We will create two data points, just for the example.
data_x = numpy.array([
    # Datapoint 1
    [
        # Input features at timestep 1
        [1, 2, 3],
        # Input features at timestep 2
        [4, 5, 6]
    ],
    # Datapoint 2
    [
        # Features at timestep 1
        [7, 8, 9],
        # Features at timestep 2
        [10, 11, 12]
    ]
])

# The desired model outputs.
# We will create two data points, just for the example.
data_y = numpy.array([
    # Datapoint 1
    [
        # Target features at timestep 1
        [101, 102, 103, 104],
        # Target features at timestep 2
        [105, 106, 107, 108]
    ],
    # Datapoint 2
    [
        # Target features at timestep 1
        [201, 202, 203, 204],
        # Target features at timestep 2
        [205, 206, 207, 208]
    ]
])

# Each input data point has 2 timesteps, each with 3 features.
# So the input shape (excluding batch_size) is (2, 3), which
# matches the shape of each data point in data_x above.
model_input = L.Input(shape=(2, 3))

# This RNN will return timesteps with 4 features each.
# Because return_sequences=True, it will output 2 timesteps, each
# with 4 features. So the output shape (excluding batch size) is
# (2, 4), which matches the shape of each data point in data_y above.
model_output = L.LSTM(4, return_sequences=True)(model_input)

# Create the model.
model = M.Model(input=model_input, output=model_output)

# You need to pick appropriate loss/optimizers for your problem.
# I'm just using these to make the example compile.
model.compile('sgd', 'mean_squared_error')

# Train
model.fit(data_x, data_y)