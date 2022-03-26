"""Base Model."""

import logging
from abc import ABC


class BasePredictionModel(ABC):
    """Init."""

    def __init__(self) -> None:
        """Init."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Init...")
