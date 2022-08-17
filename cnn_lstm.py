from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.layers import MaxPooling1D, Flatten
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.model_selection import train_test_split

#X_train.shape = (1618, 5, 7)
#CNN
model = Sequential()
model.add(Conv1D(32, kernel_size=(3,), padding='same', activation='relu', input_shape = (X_train.shape[1],1)))
model.add(Conv1D(64, kernel_size=(3,), padding='same', activation='relu'))
model.add(Conv1D(128, kernel_size=(5,), padding='same', activation='relu'))
model.add(Flatten()) #output shape: (None, 640)  

#LSTMs
#到这一步报错
#Input 0 of layer lstm_29 is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: (None, 1664)
model.add(LSTM(70, return_sequences=True, input_shape=(None,640)))
model.add(LSTM(70, return_sequences=True))
model.add(LSTM(70))
model.add(Dense(1))

#Final layers
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])