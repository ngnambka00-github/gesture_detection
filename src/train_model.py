import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from keras.layers import LSTM, Dense, Dropout, GRU
from keras.models import Sequential

from sklearn.model_selection import train_test_split

# Doc du lieu
body_swing_df = pd.read_csv("../data/BODYSWING.txt")
hand_swing_df = pd.read_csv("../data/HANDSWING.txt")

X = []
y = []
no_of_timesteps = 10

# dataset for body_swing -> label:1
dataset = body_swing_df.iloc[:, 1:].values
n_samples = len(dataset)
for i in range(no_of_timesteps, n_samples):
    X.append(dataset[(i-no_of_timesteps):i, :])
    y.append(1)

# dataset for hand_swing -> label:0
dataset = hand_swing_df.iloc[:, 1:].values
n_samples = len(dataset)
for i in range(no_of_timesteps, n_samples):
    X.append(dataset[(i-no_of_timesteps):i, :])
    y.append(0)

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

num_units_lstm = 120
model = Sequential( # model self define
    [
        LSTM(units=num_units_lstm, return_sequences=True, input_shape=(X.shape[1], X.shape[2])), 
        Dropout(0.2), 
        GRU(units=num_units_lstm, return_sequences=True), 
        Dropout(0.2), 
        GRU(units=num_units_lstm, return_sequences=True),
        Dropout(0.2),
        LSTM(units=num_units_lstm), 
        Dropout(0.2), 
        Dense(units=64, activation="relu"),
        Dropout(0.2),
        Dense(units=32, activation="relu"),
        Dense(units=1, activation="sigmoid"), # output la 1 gia tri: >0.5:body_swing, <0.5:hand_swing
    ]
)

print(model.summary())

model.compile(optimizer="adam", metrics=["accuracy"], loss="binary_crossentropy")
model.fit(X_train, y_train, epochs=16, batch_size=32, validation_data=(X_test, y_test))

model.save("../model/best_model.h5")