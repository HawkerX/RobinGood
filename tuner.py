############################################
# SUPPRESSES tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
############################################

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras_tuner
import pandas as pd
import numpy as np

######### CONSTS ############
# SEQUENCE_LENGTH determines how many past time 
# steps the model looks at to predict the next value.
SEQUENCE_LENGTH = 10
EPOCHS = 100
BATCH_SIZE = 5

#############################

# Load the Dataset
data = pd.read_csv('COCO_COLA.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)


# Preprocess 
# Use all columns and normalize them
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create Training and Testing Datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences for LSTM
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])
    return np.array(x), np.array(y)

x_train, y_train = create_sequences(train_data, SEQUENCE_LENGTH)
x_test, y_test = create_sequences(test_data, SEQUENCE_LENGTH)


# Reshape data to 3D for GRU
x_train = np.reshape(x_train, (x_train.shape[0], SEQUENCE_LENGTH, scaled_data.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], SEQUENCE_LENGTH, scaled_data.shape[1]))

"""
Assign the first layer as the shape of the data

Dropout is a regularization technique used to prevent overfitting during training.
It randomly "drops" a certain percentage of the neurons.
0.2 drops 20% of the neurons

"""
# Build the GRU Model
def call_existing_code(units, lr, dropout, dp, delta):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, scaled_data.shape[1])))
    model.add(tf.keras.layers.GRU(units = units, return_sequences=True))
    if dropout:
        model.add(tf.keras.layers.Dropout(dp)) #Dropout; look up BatchNormalization

    model.add(tf.keras.layers.GRU(units = units, return_sequences=False))
    model.add(tf.keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate = lr) # for variable learing rate
    model.compile(optimizer=optimizer, loss = tf.keras.losses.Huber(delta = delta))
    return model
    
def build_model(hp):
    units = hp.Int("units", min_value = 25, max_value = 125, step = 25)
    delta = hp.Float("delta", min_value = 1, max_value = 2.5, step = 0.5)
    dropout = hp.Boolean("dropout")
    lr = hp.Float("lr", min_value = 0.0006, max_value = 0.001, sampling = "log")
    dp = hp.Float("dp", min_value = 1e-2, max_value = 1e-1, sampling = "log")
    model = call_existing_code(units = units, lr = lr, dropout = dropout, dp = dp, delta = delta)
    return model
#Best model so far:
UNITS = 100
LR = 0.00020022 
DROPOUT = True
DP = 0.018381
DELTA = 1

#produced val_loss of: 1e-6(apporx)
"""
LOSS FUNCTIONS

mean_squared_error # overly cautious.
mean_absolute_error # less accurate predictions if there are large fluctuations
tf.keras.losses.Huber(delta=1.0) # Requires tuning a parameter
mean_absolute_percentage_error # Can be problematic if the stock price is close to zero. sensitive to extreme values.
mean_squared_logarithmic_error # less accurate predictions
Quantile Loss (Pinball Loss) # Custom Loss Function
"""


tuner = keras_tuner.BayesianOptimization(
    hypermodel = build_model,
    objective = "val_loss",
    max_trials = 25,
    beta = 2.6,
    seed = None,
    hyperparameters = None,
    tune_new_entries = True,
    allow_new_entries = True,
    max_retries_per_trial = 0,
    max_consecutive_failed_trials = 3,
    overwrite = True,
    directory = "my_dir",
    project_name = "mmodelss3"
)

tuner.search_space_summary()
tuner.search(x_train, y_train, epochs = 35, validation_data = (x_test, y_test))

models = tuner.get_best_models(num_models = 3)[0]
best_model = models
print("-------------------------")
best_model.summary()
print("-------------------------")
tuner.results_summary()