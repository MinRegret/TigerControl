# Controller class
# Author: John Hallman

from tigercontrol import error
from tigercontrol.utils.optimizers import Optimizer

# class for implementing algorithms with enforced modularity
class Controller(object):

    def plan(self, x, horizon, **kwargs):
        # returns a series of actions (a plan), given current observation x
        raise NotImplementedError

    def get_action(self, x, replan=False, horizon=None):
        """ Description: returns action based on input state x """
        raise NotImplementedError

    def update(self, **kwargs):
        # update parameters according to given loss and update rule
        raise NotImplementedError

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)
        
    def __repr__(self):
        return self.__str__()
        

""" # OLD CODE
    def get_action(self, x, replan=False, horizon=None):
        if horizon == None: horizon = self.T
        if hasattr(self, "plan_cache") and not replan:
            u = self.plan_cache.pop(0)
            if len(self.plan_cache) == 0:
                self.plan_cache = self.plan(x, horizon)
        else:
            self.plan_cache = self.plan(x, horizon)
            u = self.plan_cache.pop(0)
        return u
"""
