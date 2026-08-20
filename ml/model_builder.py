import os

os.environ.setdefault("KERAS_BACKEND", "torch")

from keras import Sequential
from keras.layers import Dense, Dropout, LSTM


def build_lstm_model(look_back: int, num_features: int) -> Sequential:
    """Replica a arquitetura LSTM definida no notebook_1.ipynb."""
    model = Sequential(
        [
            LSTM(units=50, return_sequences=True, input_shape=(look_back, num_features)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
