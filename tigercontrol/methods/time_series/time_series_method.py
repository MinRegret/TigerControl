# TimeSeriesMethod class
# Author: John Hallman

from tigercontrol import error
from tigercontrol.methods import Method

class TimeSeriesMethod(Method):
    ''' Description: class for implementing algorithms with enforced modularity '''
    def __init__(self):
        pass

    def initialize(self, **kwargs):
        pass

    def __str__(self):
    	return "<TimeSeriesMethod>"

    def __repr__(self):
        return self.__str__()

