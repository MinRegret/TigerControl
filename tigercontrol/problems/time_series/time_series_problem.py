# Problem class
# Author: John Hallman

from tigercontrol import error
from tigercontrol.problems import Problem

class TimeSeriesProblem(Problem):
    ''' Description: class for online control tests '''
    def initialize(self, **kwargs):
        ''' Description: resets problem to time 0 '''
        self.has_regressors = None
        raise NotImplementedError

    def step(self, action=None):
        ''' Description: Run one timestep of the problem's dynamics. '''
        raise NotImplementedError

