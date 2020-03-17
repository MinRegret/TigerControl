import re
import importlib
import warnings

from tigercontrol import error
from tigercontrol.utils import Spec, Registry
import copy

# This format is true today, but it's *not* an official spec.
# [username/](planner-name) - Note: Controller name must start with a capital letter!
planner_id_re = re.compile(r'^(?:[\w:-]+\/)?[A-Z]+([\w:.-]+)$')


# Have a global planner_registry
planner_registry = Registry(planner_id_re)

def planner_register(id, **kwargs):
    return planner_registry.register(id, **kwargs)

def planner_spec(id):
    return planner_registry.spec(id)

def planner(id, **kwargs):
    #return planner_registry.make(id, **kwargs)
    return planner_registry.get_class(id) # return planner class instead of single instance

