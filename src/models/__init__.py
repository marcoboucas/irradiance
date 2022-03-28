"""Init."""

import importlib
from typing import Literal

from .base_model import BasePredictionModel

def get_model(class_path: str, **kwargs) -> BasePredictionModel:
    """Get the model and return the instance."""
    module_name, class_name = class_path.rsplit(".", 1)
    my_module = importlib.import_module(f".{module_name}", package="src.models")
    class_  = getattr(my_module, class_name)
    return class_(**kwargs)