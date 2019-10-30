import re
import importlib
import warnings

from tigercontrol import error
from tigercontrol.utils import Spec, Registry

# This format is true today, but it's *not* an official spec.
# [username/](environment-name)-v(version)    environment-name is group 1, version is group 2
environment_id_re = re.compile(r'^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$')


class EnvironmentSpec(Spec):
    """A specification for a particular instance of the environment. Used
    to register the parameters for official evaluations.

    Args:
        id (str): The official environment ID
        entry_point (Optional[str]): The Python entrypoint of the environment class (e.g. module.name:Class)
        reward_threshold (Optional[int]): The reward threshold before the task is considered solved
        kwargs (dict): The kwargs to pass to the environment class
        nondeterministic (bool): Whether this environment is non-deterministic even after seeding
        tags (dict[str:any]): A set of arbitrary key-value tags on this environment, including simple property=True tags
        max_episode_steps (Optional[int]): The maximum number of steps that an episode can consist of

    Attributes:
        id (str): The official environment ID
    """
    def __str__(self):
        return "<TigerControl Environment Spec>"


class EnvironmentRegistry(Registry):
    """Register an environment by ID. IDs remain stable over time and are
    guaranteed to resolve to the same environment dynamics (or be
    desupported). The goal is that results on a particular environment
    should always be comparable, and not depend on the version of the
    code that was running.
    """
    def __str__(self):
        return "<TigerControl Environment Registry>"



# Have a global environment_registry
environment_registry = EnvironmentRegistry(environment_id_re)

def environment_register(id, **kwargs):
    return environment_registry.register(id, **kwargs)

def environment_spec(id):
    return environment_registry.spec(id)

def environment(id, **kwargs):
    return environment_registry.make(id, **kwargs)



