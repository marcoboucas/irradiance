"""Sarima model."""


import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from src.models.base_model import BasePredictionModel


class SarimaPredictionModel(BasePredictionModel):
    """Init."""

    def __init__(self) -> None:
        """Init."""
        super().__init__()

    def fit(self, *, train_data: np.ndarray) -> None:
        """Fit the model."""
        self.logger.info("Fit the model...")
        self.model = ARIMA(train_data, order=(2, 2, 2))
        self.model_fit = self.model.fit()
        self.logger.info("Model fit.")
