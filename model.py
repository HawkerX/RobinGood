
import tensorflow as tf

#Data preparation goes here: x = input, y = output

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape = (16,4))) #arbitrary shape (time steps, num of features/step)
model.add(tf.keras.layers.GRU(units = 50, return_sequences = True))#GRUs are LSTM RNNs and faster; TBD: activation & shape
model.add(tf.keras.layers.Dropout(0.8))#Dropout; look up BatchNormalization
model.add(tf.keras.layers.GRU(units = 25, return_sequences = False))
model.add(tf.keras.layers.Dense(1)) #predicting the future price
model.compile(optimizer = "adam", loss = "mean_squared_error") #good enough for now

## OPTIONAL line to print the current model
model.summary()

#>>>train the model