# Problem class
# Author: John Hallman

from tigercontrol.problems import Problem

class ControlProblem(Problem):
    ''' Description: class for online control tests '''

    def initialize(self, **kwargs):
        ''' Description: resets problem to time 0 '''
        raise NotImplementedError

    def step(self, action=None):
        ''' Description: run one timestep of the problem's dynamics. '''
        raise NotImplementedError
