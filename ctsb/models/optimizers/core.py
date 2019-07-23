"""
Core class for optimizers 
"""

class Optimizer():
    ''' Description: the core class for optimizers '''
    def __init__(self, loss, learning_rate, params_dict):
        self.loss = loss
        self.lr = learning_rate
        self.params_dict = params_dict

    def update(x, y, pred, model_params):
        raise NotImplementedError
