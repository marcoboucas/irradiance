"""Base Model."""

import logging
from abc import ABC, abstractmethod

import numpy as np

class BasePredictionModel(ABC):
    """Init."""

    def __init__(self) -> None:
        """Init."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Init...")

    @abstractmethod
    def predict(
        self, initial_values: np.ndarray, how_many: int = 400, verbose: bool = False
    ) -> np.ndarray:
        """Predict."""


    def fit(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> None:
        """Train the model."""
        pass