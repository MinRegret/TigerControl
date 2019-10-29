# Method class
# Author: John Hallman

from tigercontrol import error
from tigercontrol.methods import Method

class ControlMethod(Method):
    ''' Description: class for implementing algorithms with enforced modularity '''
    def __init__(self):
        pass

    def initialize(self, **kwargs):
        pass


    def __str__(self):
    	return "<ControlMethod>"

    def __repr__(self):
        return self.__str__()