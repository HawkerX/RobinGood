############################################
# SUPPRESSES tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
############################################

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import numpy as np

######### CONSTS ############
# SEQUENCE_LENGTH determines how many past time 
# steps the model looks at to predict the next value.
SEQUENCE_LENGTH = 10
EPOCHS = 50
BATCH_SIZE = 5

# TODO Optimize these variables
DROPOUT = 0.4 # 0.8
LEARNING_RATE = 0.0003 # 0.001
DELTA = 1 # for Huber Loss Function (avg = 1 to 2)
PATIENCE = 10 # 5

#############################

# Load the Dataset
data = pd.read_csv('GOOGL_Training_Data.csv')
data = data[['Date', 'Open']]
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)


# Preprocess # TODO might varry depending on the acctual data we use
# Use only the 'Open' column and normalize it
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
        x.append(data[i:i + seq_length, 0])
        y.append(data[i + seq_length, 0])
    return np.array(x), np.array(y)

x_train, y_train = create_sequences(train_data, SEQUENCE_LENGTH)
x_test, y_test = create_sequences(test_data, SEQUENCE_LENGTH)


# Reshape data to 3D for GRU
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

"""
Assign the first layer as the shape of the data

Dropout is a regularization technique used to prevent overfitting during training.
It randomly "drops" a certain percentage of the neurons.
0.2 drops 20% of the neurons

"""
# Build the GRU Model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(x_train.shape[1], 1))) # Input Shap is the train data shape
model.add(tf.keras.layers.GRU(50, return_sequences=True))
model.add(tf.keras.layers.Dropout(DROPOUT)) #Dropout; look up BatchNormalization

model.add(tf.keras.layers.GRU(50, return_sequences=False))
model.add(tf.keras.layers.Dropout(DROPOUT))
model.add(tf.keras.layers.Dense(1))

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE) # for variable learing rate
model.compile(optimizer=optimizer, loss="mean_squared_error") 
"""
LOSS FUNCTIONS

mean_squared_error # overly cautious.
mean_absolute_error # less accurate predictions if there are large fluctuations
tf.keras.losses.Huber(delta=1.0) # Requires tuning a parameter
mean_absolute_percentage_error # Can be problematic if the stock price is close to zero. sensitive to extreme values.
mean_squared_logarithmic_error # less accurate predictions
Quantile Loss (Pinball Loss) # Custom Loss Function
"""

model.summary()  # Optional to print the model structure

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=PATIENCE, restore_best_weights=True)
model.fit(
    x_train, y_train, 
    validation_split=0.1,
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS, 
    callbacks=[early_stopping])

# make predictions
preds = model.predict(x_test)
preds = scaler.inverse_transform(preds)

########### PLOT ###############
# Plot the actual vs predicted values
plt.figure(figsize=(14,7))

# Plotting the actual stock prices (true values)
plt.plot(data.index[train_size + SEQUENCE_LENGTH:], scaler.inverse_transform(test_data[SEQUENCE_LENGTH:]), label='Actual', color='green')

# Plotting the predicted stock prices
plt.plot(data.index[train_size + SEQUENCE_LENGTH:], preds, label='Predicted', color='red')

# Adding titles and labels
plt.title('Stock Price Prediction vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()

# Show the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

###################################
