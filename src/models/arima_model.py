"""Sarima model."""


import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

from src.models.base_model import BasePredictionModel


class ArimaPredictionModel(BasePredictionModel):
    """Init."""

    def __init__(self, batch_pred_size: int = 10) -> None:
        """Init."""
        super().__init__()
        self.batch_pred_size = batch_pred_size

    def fit(self, **kwargs) -> None:
        """Fit the model."""
        self.logger.info("The model is trained during the prediction at each step...")

    def predict(self, initial_values: np.ndarray, how_many: int = 400, verbose: bool = False) -> np.ndarray:
        """Predict using ARIMA."""
        input_size = initial_values.shape[0]
        x_final = np.zeros((initial_values.shape[0]+how_many,))
        x_final[: input_size] = initial_values[-input_size :]
        for t in tqdm(range(input_size, x_final.shape[0]-1, self.batch_pred_size) ,disable=not verbose):
            model = ARIMA(x_final[:t], order=(1, 1, 1))
            model_fit = model.fit()
            output = model_fit.forecast(10)
            x_final[t:t+self.batch_pred_size] = output[0]
        return x_final
