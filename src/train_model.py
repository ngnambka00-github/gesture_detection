import numpy as np
import pandas as pd
import os

import keras
from keras.layers import LSTM, Dense, Dropout, GRU
from keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from hydra import initialize, compose
from omegaconf import OmegaConf

import warnings
warnings.filterwarnings('ignore')

with initialize(config_path="../configs/"):
    data_cfg = compose(config_name="data_path")
data_cfg = OmegaConf.create(data_cfg)
HOME_PATH = "../"

body_swing_data_path = os.path.join(HOME_PATH, data_cfg.data.body_swing)
hand_left_swing_data_path = os.path.join(HOME_PATH, data_cfg.data.hand_left_swing)
hand_right_swing_data_path = os.path.join(HOME_PATH, data_cfg.data.hand_right_swing)
hand_two_swing_data_path = os.path.join(HOME_PATH, data_cfg.data.hand_two_swing)

best_model_path = os.path.join(HOME_PATH, data_cfg.model.best_model)
checkpoint_path = os.path.join(HOME_PATH, data_cfg.model.checkpoint)

# Doc du lieu
body_swing_df = pd.read_csv(body_swing_data_path)
hand_left_swing_df = pd.read_csv(hand_left_swing_data_path)
hand_right_swing_df = pd.read_csv(hand_right_swing_data_path)
hand_two_swing_df = pd.read_csv(hand_two_swing_data_path)

dataset_list = [body_swing_df, hand_left_swing_df, hand_right_swing_df, hand_two_swing_df]
X = []
y = []
no_of_timesteps = 10

for idx, ds in enumerate(dataset_list):
    # dataset for body_swing, hand_left_swing, hand_right_swing, hand_two_swing -> label:0, 1, 2, 3
    dataset = ds.iloc[:,1:].values
    n_samples = len(dataset)
    for i in range(no_of_timesteps, n_samples):
        X.append(dataset[(i-no_of_timesteps):i, :])
        y.append(idx)

enc = OneHotEncoder()
X, y = np.array(X), np.array(y)
y = enc.fit_transform(y.reshape(-1, 1)).toarray()

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
        Dense(units=4, activation="softmax"), # output la 1 gia tri: >0.5:body_swing, <0.5:hand_swing
    ]
)

print(model.summary())

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath= checkpoint_path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.compile(optimizer="adam", metrics=["accuracy"], loss="binary_crossentropy")
model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_data=(X_test, y_test), 
    callbacks=[model_checkpoint_callback]
)

model.load_weights(checkpoint_path)
model.save(best_model_path)