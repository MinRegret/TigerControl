import re
import importlib
import warnings

from ctsb import error
from ctsb.utils import Spec, Registry
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
        return "<CTSB Model Spec>"


class ModelRegistry(Registry):
    """Register an model by ID. IDs remain stable over time and can
    be called via ctsb.model("ID").
    """

    def __str__(self):
        return "<CTSB Model Registry>"

    def make(self, path, **kwargs):
        """
        Args: 
            path(string): id of object in registry
            kwargs(dict): The kwargs to pass to the object class
        Returns:
            object instance
        """
        if path in self.custom:
            return self.custom[path]()

        spec = self.spec(path)
        obj = spec.make(**kwargs)
        return obj

    # register a custom model class
    def register_custom(self, id, custom_class):
        self.custom[id] = custom_class


# Have a global model_registry
model_registry = ModelRegistry(model_id_re)

def model_register(id, **kwargs):
    return model_registry.register(id, **kwargs)

def model(id, **kwargs):
    return model_registry.make(id, **kwargs)

def model_spec(id):
    return model_registry.spec(id)


