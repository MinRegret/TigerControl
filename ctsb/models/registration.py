import re
import importlib
import warnings

from ctsb import error, logger
from ctsb.utils import Spec, Registry

# This format is true today, but it's *not* an official spec.
# [username/](model-name)-v(version)    model-name is group 1, version is group 2
model_id_re = re.compile(r'^(?:[\w:-]+\/)?([\w:.-]+)$')


class ModelSpec(Spec):
    """A specification for a particular instance of the model. Used
    to register the parameters for official evaluations.

    Args:
        id (str): The official model ID
        entry_point (Optional[str]): The Python entrypoint of the model class (e.g. module.name:Class)
        reward_threshold (Optional[int]): The reward threshold before the task is considered solved
        kwargs (dict): The kwargs to pass to the model class
        nondeterministic (bool): Whether this model is non-deterministic even after seeding
        tags (dict[str:any]): A set of arbitrary key-value tags on this model, including simple property=True tags
        max_episode_steps (Optional[int]): The maximum number of steps that an episode can consist of

    Attributes:
        id (str): The official model ID
    """
    def __str__(self):
        return "Model"



class ModelRegistry(Registry):
    """Register an model by ID. IDs remain stable over time and are
    guaranteed to resolve to the same model dynamics (or be
    desupported). The goal is that results on a particular model
    should always be comparable, and not depend on the version of the
    code that was running.
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

warn_once = True

