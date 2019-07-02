# custom method registration tools

from ctsb.models.custom import CustomModel
from ctsb.models.registration import model_registry
from ctsb import error
import copy


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

