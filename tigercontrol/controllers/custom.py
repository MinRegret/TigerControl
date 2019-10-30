# Controller class
# Author: John Hallman

from tigercontrol import error
from tigercontrol.controllers.core import Controller
from tigercontrol.controllers.registration import controller_registry
from tigercontrol.utils.optimizers import Optimizer


# class for implementing algorithms with enforced modularity
class CustomController(object):
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

# verifies that a given class has the necessary minimum controller controllers
def verify_valid_controller(controller_class):
    assert issubclass(controller_class, CustomController)
    for f in ['initialize', 'predict', 'update']:
        if not callable(getattr(controller_class, f, None)):
            raise error.InvalidClass("CustomController is missing required controller \'{}\'".format(f))

# global custom controller controller
def register_custom_controller(custom_controller_class, custom_controller_id):
    assert type(custom_controller_id) == str
    verify_valid_controller(custom_controller_class)

    controller_registry.register_custom(
        id=custom_controller_id,
        custom_class=custom_controller_class,
    )

