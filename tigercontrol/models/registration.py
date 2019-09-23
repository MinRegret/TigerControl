import re
import importlib
import warnings

from tigercontrol import error
from tigercontrol.utils import Spec, Registry
import copy

# This format is true today, but it's *not* an official spec.
# [username/](model-name) - Note: Model name must start with a capital letter!
model_id_re = re.compile(r'^(?:[\w:-]+\/)?[A-Z]+([\w:.-]+)$')


class ModelSpec(Spec):
    """A specification for a particular instance of the model. Used
    to register the parameters for official evaluations.

    Args:
        id (str): The official model ID
        entry_point (Optional[str]): The Python entrypoint of the problem class (e.g. module.name:Class)
        kwargs (dict): The kwargs to pass to the model class

    Attributes:
        id (str): The official problem ID
    """
    def __str__(self):
        return "<TigerControl Model Spec>"


class ModelRegistry(Registry):
    """Register an model by ID. IDs remain stable over time and can
    be called via tigercontrol.model("ID").
    """

    def __str__(self):
        return "<TigerControl Model Registry>"

# Have a global model_registry
model_registry = ModelRegistry(model_id_re)

def model_register(id, **kwargs):
    return model_registry.register(id, **kwargs)

def model(id, **kwargs):
    return model_registry.make(id, **kwargs)

def model_spec(id):
    return model_registry.spec(id)


