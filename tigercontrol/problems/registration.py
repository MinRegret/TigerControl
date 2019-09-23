import re
import importlib
import warnings

from tigercontrol import error
from tigercontrol.utils import Spec, Registry

# This format is true today, but it's *not* an official spec.
# [username/](problem-name)-v(version)    problem-name is group 1, version is group 2
problem_id_re = re.compile(r'^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$')


class ProblemSpec(Spec):
    """A specification for a particular instance of the problem. Used
    to register the parameters for official evaluations.

    Args:
        id (str): The official problem ID
        entry_point (Optional[str]): The Python entrypoint of the problem class (e.g. module.name:Class)
        reward_threshold (Optional[int]): The reward threshold before the task is considered solved
        kwargs (dict): The kwargs to pass to the problem class
        nondeterministic (bool): Whether this problem is non-deterministic even after seeding
        tags (dict[str:any]): A set of arbitrary key-value tags on this problem, including simple property=True tags
        max_episode_steps (Optional[int]): The maximum number of steps that an episode can consist of

    Attributes:
        id (str): The official problem ID
    """
    def __str__(self):
        return "<TigerControl Problem Spec>"


class ProblemRegistry(Registry):
    """Register an problem by ID. IDs remain stable over time and are
    guaranteed to resolve to the same problem dynamics (or be
    desupported). The goal is that results on a particular problem
    should always be comparable, and not depend on the version of the
    code that was running.
    """
    def __str__(self):
        return "<TigerControl Problem Registry>"



# Have a global problem_registry
problem_registry = ProblemRegistry(problem_id_re)

def problem_register(id, **kwargs):
    return problem_registry.register(id, **kwargs)

def problem_spec(id):
    return problem_registry.spec(id)

def problem(id, **kwargs):
    return problem_registry.make(id, **kwargs)



