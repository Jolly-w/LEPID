from .base import BaseSelectionStrategy, StaticBaseSelectionStrategy
import random
import numpy as np


class RandomSelection(StaticBaseSelectionStrategy):
    def __init__(self, random_state=None):
        super().__init__()

    # def utility(self, X, clf, **kwargs):
    def utility(self, **kwargs):
        output = random.random()
        return output
