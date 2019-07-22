# Model class
# Author: John Hallman

from ctsb import error
from ctsb.models import Model

class TimeSeriesModel(Model):
    ''' Description: class for implementing algorithms with enforced modularity '''
    def __init__(self):
        pass

    def initialize(self, predict=lambda params, x: x, params=None, update=lambda params, x: params):
        ''' Description: initializes model parameters 
        
            Args: 
                predict : predict function
                params : hyperparameters of the model
                update : update function
        '''
        assert type(predict) == type(lambda x: None) # class function
        assert type(update) == type(lambda x: None) # class function

        self._predict = predict
        self._params = params
        self._update = update


