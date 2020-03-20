import re
import importlib
import warnings

from tigercontrol import error
from tigercontrol.utils import Spec, Registry

# This format is true today, but it's *not* an official spec.
# [username/](environment-name)-v(version)    environment-name is group 1, version is group 2
environment_id_re = re.compile(r'^(?:[\w:-]+\/)?([\w:.-]+)$')


# Have a global environment_registry
environment_registry = Registry(environment_id_re)

def environment_register(id, **kwargs):
    return environment_registry.register(id, **kwargs)

def environment_spec(id):
    return environment_registry.spec(id)

def environment(id, **kwargs):
    return environment_registry.make(id, **kwargs)
    # return environment_registry.get_class(id) # return env class instead of single instance



