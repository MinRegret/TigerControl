import re
import importlib
import warnings

from tigercontrol import error
from tigercontrol.utils import Spec, Registry
import copy

# This format is true today, but it's *not* an official spec.
# [username/](controller-name) - Note: Controller name must start with a capital letter!
controller_id_re = re.compile(r'^(?:[\w:-]+\/)?[A-Z]+([\w:.-]+)$')


# Have a global controller_registry
controller_registry = Registry(controller_id_re)

def controller_register(id, **kwargs):
    return controller_registry.register(id, **kwargs)

def controller_spec(id):
    return controller_registry.spec(id)

def controller(id, **kwargs):
    #return controller_registry.make(id, **kwargs)
    return controller_registry.get_class(id) # return controller class instead of single instance

