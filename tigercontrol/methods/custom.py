# Method class
# Author: John Hallman

from tigercontrol import error
from tigercontrol.methods import Method
from tigercontrol.methods.registration import method_registry
from tigercontrol.methods.optimizers import Optimizer


# class for implementing algorithms with enforced modularity
class CustomMethod(object):
    def __init__(self):
        pass
    
    def _store_optimizer(self, optimizer, pred):
        if isinstance(optimizer, Optimizer):
            optimizer.set_predict(pred)
            self.optimizer = optimizer
            return
        if issubclass(optimizer, Optimizer):
            self.optimizer = optimizer(pred=pred)
            return
        raise error.InvalidInput("Optimizer input cannot be stored")

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

# verifies that a given class has the necessary minimum method methods
def verify_valid_method(method_class):
    assert issubclass(method_class, CustomMethod)
    for f in ['initialize', 'predict', 'update']:
        if not callable(getattr(method_class, f, None)):
            raise error.InvalidClass("CustomMethod is missing required method \'{}\'".format(f))

# global custom method method
def register_custom_method(custom_method_class, custom_method_id):
    assert type(custom_method_id) == str
    verify_valid_method(custom_method_class)

    method_registry.register_custom(
        id=custom_method_id,
        custom_class=custom_method_class,
    )

