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

    def update(self, rule=None):
        # update parameters according to given loss and update rule
        raise NotImplementedError

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


# class for implementing algorithms with enforced modularity
class CustomModel(Model):

    def __init__(self):
        pass

    def initialize(self, predict, params=None, update=lambda x: x):
        # initializes model parameters
        assert type(predict) == type(lambda x: None) # class function
        assert type(update) == type(lambda x: None) # class function

        self.predict = predict
        self.params = None
        self.update = update

    def step(self, **kwargs):
        # run one timestep of the model in its environment
        raise NotImplementedError

    def predict(self, **kwargs):
        # returns model prediction for given input
        return self.predict(**kwargs)

    def update(self, **kwargs):
        # update parameters according to given loss and update rule
        return self.update(**kwargs)



