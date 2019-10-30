# Method class
# Author: John Hallman

from tigercontrol import error
from tigercontrol.methods.optimizers import Optimizer

# class for implementing algorithms with enforced modularity
class Method(object):

    def initialize(self, **kwargs):
        # initializes method parameters
        raise NotImplementedError

    def get_acttion(self, **kwargs):
        # returns action u given input observation x
        raise NotImplementedError

    def plan(self, **kwargs):
        # returns a series of actions (a plan), given current observation x
        raise NotImplementedError

    def update(self, **kwargs):
        # update parameters according to given loss and update rule
        raise NotImplementedError

    def _store_optimizer(self, optimizer, pred):
        if isinstance(optimizer, Optimizer):
            optimizer.set_predict(pred)
            self.optimizer = optimizer
            return
        if issubclass(optimizer, Optimizer):
            self.optimizer = optimizer(pred=pred)
            return
        raise error.InvalidInput("Optimizer input cannot be stored")

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)
        
    def __repr__(self):
        return self.__str__()
        