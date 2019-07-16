# Problem class
# Author: John Hallman

from ctsb import error
from ctsb.problems import Problem
from ctsb.problems.registration import problem_registry


# class for implementing algorithms with enforced modularity
class CustomProblem(object):

    def __init__(self):
        pass

    # Note: these functions MUST be implemented by the user for CustomProblem to work
    """
    def initialize(self, **kwargs):
        # initializes problem parameters
        raise NotImplementedError

    def predict(self, x):
        # returns problem prediction for given input
        raise NotImplementedError

    def update(self, **kwargs):
        # update parameters according to given loss and update rule
        raise NotImplementedError
    """