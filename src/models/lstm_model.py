"""LSTM model."""

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM
import tensorflow as tf
from tqdm import tqdm

from src.models.base_model import BasePredictionModel


class LSTMPredictionModel(BasePredictionModel):
    """LSTM Model."""

    def __init__(self) -> None:
        """Init."""
        super().__init__()
        self.input_size = 1

    def fit(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        plot_history: bool = False
    ) -> None:
        """Fit the model."""
        self.logger.info("Fitting the model...")

        self.input_size = 1 if len(X_train.shape) == 1 else X_train.shape[1]
        self.model = self.get_model(input_size=self.input_size)
        self.model.compile(
            loss="mse",
            optimizer="adam",
        )
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=0,
                    patience=2,
                    verbose=0,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=False,
                )
            ],
        )
        self.logger.info("Model fitted.")
        if plot_history:
            plt.plot(history.epoch, history.history["loss"], label="Training loss")
            plt.plot(history.epoch, history.history["val_loss"], label="Validation loss")

            plt.legend()
            plt.show()
        self.logger.info("Model fit.")

    def get_model(self, input_size: int = 1, hidden_size: int = 50) -> Sequential:
        """Get the model."""
        model = Sequential()
        model.add(Input(shape=(input_size, 1)))
        model.add(LSTM(hidden_size, input_shape=(input_size,), return_sequences=True))
        model.add(
            LSTM(hidden_size, input_shape=(hidden_size,))
        )  # (1, loop_back) # (1, input_sequence_length)
        model.add(Dense(10, activation="relu"))
        model.add(Dense(1))

        return model

    def predict(
        self, initial_values: np.ndarray, how_many: int = 400, verbose: bool = False
    ) -> np.ndarray:
        """Predict."""
        x_final = np.zeros((1, how_many))
        x_final[0, : self.input_size] = initial_values[-self.input_size :]
        for i in tqdm(range(self.input_size, how_many), disable=not verbose):
            x_final[0, i] = self.model.predict(x_final[:, i - self.input_size : i])
        return x_final[0]