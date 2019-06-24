import re
import importlib
import warnings

from ctsb import error, logger
from ctsb.utils import Spec, Registry

# This format is true today, but it's *not* an official spec.
# [username/](model-name) - Note: Model name must start with a capital letter!
model_id_re = re.compile(r'^(?:[\w:-]+\/)?[A-Z]+([\w:.-]+)$')


class ModelSpec(Spec):
    # Spec class for pre-implemented Models
    def __str__(self):
        return "Model"


class ModelRegistry(Registry):
    """Register an model by ID. IDs remain stable over time and can
    be called via ctsb.model("ID").
    """
    def __str__(self):
        return "Model"



# Have a global model_registry
model_registry = ModelRegistry(model_id_re)

def model_register(id, **kwargs):
    return model_registry.register(id, **kwargs)

def model(id, **kwargs):
    return model_registry.make(id, **kwargs)

def model_spec(id):
    return model_registry.spec(id)


