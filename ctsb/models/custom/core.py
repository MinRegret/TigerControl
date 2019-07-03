# Model class
# Author: John Hallman

from ctsb import error
from ctsb.models import Model
from ctsb.models.registration import model_registry


# class for implementing algorithms with enforced modularity
class CustomModel(object):

    def __init__(self):
        pass

    # Note: these functions MUST be implemented by the user for CustomModel to work
    """
    def initialize(self, **kwargs):
        # initializes model parameters
        raise NotImplementedError

    def predict(self, x):
        # returns model prediction for given input
        raise NotImplementedError

    def update(self, **kwargs):
        # update parameters according to given loss and update rule
        raise NotImplementedError
    """