# Problem class
# Author: John Hallman

from ctsb import error
from ctsb.problems import Problem

# class for online control tests
class TimeSeriesProblem(Problem):
    spec = None

    def __init__(self):
        self.initialized = False

    def initialize(self, **kwargs):
        # resets problem to time 0
        raise NotImplementedError

    def step(self, action=None):
        #Run one timestep of the problem's dynamics. 
        raise NotImplementedError

    def close(self):
        # closes the problem and returns used memory
        pass

    def help(self):
        # prints information about this class and its methods
        raise NotImplementedError

    def __str__(self):
        if self.spec is None:
            return '<{} instance> call object help() method for info'.format(type(self).__name__)
        else:
            return '<{}<{}>> call object help() method for info'.format(type(self).__name__, self.spec.id)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        # propagate exception
        return False

