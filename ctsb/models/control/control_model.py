# Model class
# Author: John Hallman

from ctsb import error
from ctsb.models import Model

# class for implementing algorithms with enforced modularity
class ControlModel(Model):

    def __init__(self):
        pass

    def initialize(self, predict=lambda params, x: x, params=None, update=lambda params, x: params):
        # initializes model parameters
        assert type(predict) == type(lambda x: None) # class function
        assert type(update) == type(lambda x: None) # class function

        self._predict = predict
        self._params = params
        self._update = update


