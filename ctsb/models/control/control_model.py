# Model class
# Author: John Hallman

from ctsb import error
from ctsb.models import Model


# class for implementing algorithms with enforced modularity
class ControlModel(ctsb.Model):

    def __init__(self):
        pass

    def initialize(self, predict=lambda params, x: x, params=None, update=lambda params, x: params):
        # initializes model parameters
        assert type(predict) == type(lambda x: None) # class function
        assert type(update) == type(lambda x: None) # class function

        self._predict = predict
        self._params = params
        self._update = update

    def step(self, **kwargs):
        # run one timestep of the model in its environment
        raise NotImplementedError

    def predict(self, x):
        # returns model prediction for given input
        return self._predict(self._params, x)

    def update(self, args):
        # update parameters according to given loss and update rule
        self._params = self._update(self._params, args)

    def get_params(self):
        return self._params

