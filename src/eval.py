"""Evaluation function."""

from abc import ABC, abstractmethod
import numpy as np

class BaseEvaluator(ABC):
    """Base Evaluator."""
    
    @abstractmethod
    def __call__(self, *, pred: np.ndarray, truth: np.ndarray) -> float:
        """Compute the evaluation metric."""


class MSE(BaseEvaluator):
    """Mean Squared Error."""
    
    def __call__(self, *, pred: np.ndarray, truth: np.ndarray) -> float:
        """Compute the evaluation metric."""
        return np.mean((pred - truth)**2)