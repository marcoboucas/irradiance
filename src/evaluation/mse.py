"""MSE evaluator."""
import numpy as np

from .base_eval import BaseEvaluator


class MSE(BaseEvaluator):
    """Mean Squared Error."""

    def __call__(self, *, pred: np.ndarray, truth: np.ndarray) -> float:
        """Compute the evaluation metric."""
        return np.mean((pred - truth) ** 2)
