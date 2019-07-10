# Model class
# Author: John Hallman

from ctsb import error

# class for implementing algorithms with enforced modularity
class Model(object):
    spec = None

    def initialize(self, **kwargs):
        # initializes model parameters
        raise NotImplementedError

    def step(self, **kwargs):
        # run one timestep of the model in its environment
        raise NotImplementedError

    def predict(self, x=None):
        # returns model prediction for given input
        raise NotImplementedError

    def update(self, **kwargs):
        # update parameters according to given loss and update rule
        raise NotImplementedError

    def help(self):
        # prints information about this class and its methods
        raise NotImplementedError

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        # propagate exception
        return False



