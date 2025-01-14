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
    # model.add(tf.keras.layers.Input(shape=(x_train.shape[1], 1))) # Input Shap is the train data shape
    model.add(tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, scaled_data.shape[1])))
    model.add(tf.keras.layers.GRU(units = 100, return_sequences=True))
    if dropout:
        model.add(tf.keras.layers.Dropout(dp)) #Dropout; look up BatchNormalization

    model.add(tf.keras.layers.GRU(units = 100, return_sequences=False))
    model.add(tf.keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate = lr) # for variable learing rate
    model.compile(optimizer=optimizer, loss = tf.keras.losses.Huber(delta = delta))
    return model
#lr = 0.0001/0.00020022
#Optimized hyperparameters:
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



model = call_existing_code(units = UNITS, lr = LR, dropout = DROPOUT, dp = DP, delta = DELTA)

model.summary()  # Optional to print the model structure

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(
    x_train, y_train, 
    validation_split = 0.1,
    batch_size = BATCH_SIZE, 
    epochs = EPOCHS, 
    callbacks = [early_stopping])
# make predictions
preds = model.predict(x_test)


print("Shape of scaled_data:", scaled_data.shape)
print("Shape of preds:", preds.shape)

# Create a dummy array with the same number of features as scaled_data
dummy = np.zeros((preds.shape[0], scaled_data.shape[1]))

# Assuming you're predicting the first feature (index 0)
dummy[:, 0] = preds.flatten()

# Apply inverse_transform to the dummy array
preds_inverse = scaler.inverse_transform(dummy)

# Extract the predicted feature (first column)
preds = preds_inverse[:, 0]

print("Shape of preds after processing:", preds.shape)
actual = scaler.inverse_transform(test_data[SEQUENCE_LENGTH:])[:, 0]
plt.figure(figsize=(14,7))

# Plotting the actual stock prices (true values)
plt.plot(data.index[train_size + SEQUENCE_LENGTH:], actual, label='Actual', color='green')

# Plotting the predicted stock prices
plt.plot(data.index[train_size + SEQUENCE_LENGTH:], preds, label='Predicted', color='red')

# Adding titles and labels
plt.title('Stock Price Prediction (Open) vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price (Open)')
plt.legend()

# Show the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

###################################
