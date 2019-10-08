import re
import importlib
import warnings

from tigercontrol import error
from tigercontrol.utils import Spec, Registry
import copy

# This format is true today, but it's *not* an official spec.
# [username/](method-name) - Note: Method name must start with a capital letter!
method_id_re = re.compile(r'^(?:[\w:-]+\/)?[A-Z]+([\w:.-]+)$')


class MethodSpec(Spec):
    """A specification for a particular instance of the method. Used
    to register the parameters for official evaluations.

    Args:
        id (str): The official method ID
        entry_point (Optional[str]): The Python entrypoint of the problem class (e.g. module.name:Class)
        kwargs (dict): The kwargs to pass to the method class

    Attributes:
        id (str): The official problem ID
    """
    def __str__(self):
        return "<TigerControl Method Spec>"


class MethodRegistry(Registry):
    """Register an method by ID. IDs remain stable over time and can
    be called via tigercontrol.method("ID").
    """

    def __str__(self):
        return "<TigerControl Method Registry>"

# Have a global method_registry
method_registry = MethodRegistry(method_id_re)

def method_register(id, **kwargs):
    return method_registry.register(id, **kwargs)

def method(id, **kwargs):
    return method_registry.make(id, **kwargs)

def method_spec(id):
    return method_registry.spec(id)


