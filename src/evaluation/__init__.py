"""Evaluators."""


import importlib
from typing import Literal

from .base_eval import BaseEvaluator


def get_evaluator(class_path: str) -> BaseEvaluator:
    """Get the model and return the instance."""
    module_name, class_name = class_path.rsplit(".", 1)
    my_module = importlib.import_module(f".{module_name}", package="src.evaluation")
    class_ = getattr(my_module, class_name)
    return class_()
