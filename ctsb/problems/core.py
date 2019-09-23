# Problem class
# Author: John Hallman

from tigercontrol import error

# class for online control tests
class Problem(object):
    spec = None

    def __init__(self):
        self.initialized = False

    def initialize(self, **kwargs):
        ''' Description: resets problem to time 0 '''
        raise NotImplementedError

    def step(self, action=None):
        ''' Description: run one timestep of the problem's dynamics. '''
        raise NotImplementedError

    def close(self):
        ''' Description: closes the problem and returns used memory '''
        pass

    def help(self):
        ''' Description: prints information about this class and its methods '''
        raise NotImplementedError

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def __enter__(self):
        return self
