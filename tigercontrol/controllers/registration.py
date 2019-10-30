import re
import importlib
import warnings

from tigercontrol import error
from tigercontrol.utils import Spec, Registry
import copy

# This format is true today, but it's *not* an official spec.
# [username/](controller-name) - Note: Controller name must start with a capital letter!
controller_id_re = re.compile(r'^(?:[\w:-]+\/)?[A-Z]+([\w:.-]+)$')


class ControllerSpec(Spec):
    """A specification for a particular instance of the controller. Used
    to register the parameters for official evaluations.

    Args:
        id (str): The official controller ID
        entry_point (Optional[str]): The Python entrypoint of the environment class (e.g. module.name:Class)
        kwargs (dict): The kwargs to pass to the controller class

    Attributes:
        id (str): The official environment ID
    """
    def __str__(self):
        return "<TigerControl Controller Spec>"


class ControllerRegistry(Registry):
    """Register an controller by ID. IDs remain stable over time and can
    be called via tigercontrol.controllers("ID").
    """

    def __str__(self):
        return "<TigerControl Controller Registry>"

# Have a global controller_registry
controller_registry = ControllerRegistry(controller_id_re)

def controller_register(id, **kwargs):
    return controller_registry.register(id, **kwargs)

def controller(id, **kwargs):
    return controller_registry.make(id, **kwargs)

def controller_spec(id):
    return controller_registry.spec(id)


