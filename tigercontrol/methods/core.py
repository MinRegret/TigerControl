# Method class
# Author: John Hallman

from tigercontrol import error
from tigercontrol.methods.optimizers import Optimizer

# class for implementing algorithms with enforced modularity
class Method(object):

    def initialize(self, T, **kwargs):
        # initializes method parameters
        self.T = T
        self.plan_cache = []
        self.plan_pointer = 0
        raise NotImplementedError

    def get_action(self, **kwargs):
        # returns action u given input observation x
        raise NotImplementedError

    def plan(self, x, horizon, **kwargs):
        # returns a series of actions (a plan), given current observation x
        raise NotImplementedError

    def get_action(self, x, replan=False, horizon=self.T):
        if replan or self.plan_pointer == len(self.plan_cache):
            self.plan_cache = self.plan(x, horizon)
            self.plan_pointer = 0
        action = self.plan_cache[self.plan_pointer]
        self.plan_pointer += 1
        return action


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
        