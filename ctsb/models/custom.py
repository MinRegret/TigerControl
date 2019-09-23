# Model class
# Author: John Hallman

from tigercontrol import error
from tigercontrol.models import Model
from tigercontrol.models.registration import model_registry
from tigercontrol.models.optimizers import Optimizer


# class for implementing algorithms with enforced modularity
class CustomModel(object):
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

# verifies that a given class has the necessary minimum model methods
def verify_valid_model(model_class):
    assert issubclass(model_class, CustomModel)
    for f in ['initialize', 'predict', 'update']:
        if not callable(getattr(model_class, f, None)):
            raise error.InvalidClass("CustomModel is missing required method \'{}\'".format(f))

# global custom model method
def register_custom_model(custom_model_class, custom_model_id):
    assert type(custom_model_id) == str
    verify_valid_model(custom_model_class)

    model_registry.register_custom(
        id=custom_model_id,
        custom_class=custom_model_class,
    )

