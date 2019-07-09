import re
import importlib
import warnings

from ctsb import error
from ctsb.utils import Spec, Registry
import pybullet_envs

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
        return "<CTSB Problem Spec>"


class ProblemRegistry(Registry):
    """Register an problem by ID. IDs remain stable over time and are
    guaranteed to resolve to the same problem dynamics (or be
    desupported). The goal is that results on a particular problem
    should always be comparable, and not depend on the version of the
    code that was running.
    """
    def __str__(self):
        return "<CTSB Problem Registry>"

    def __init__(self, regexp):
        self.specs = {}
        self.regexp = regexp
        self.pybullet_passed = []
        self.pybullet_failed = []
        for name in [name.split('- ')[-1] for name in pybullet_envs.getList()]:
            try:
                #env = pybullet_envs.make(name) # TODO: decomment eventually, figure out how
                #env.close()
                self.pybullet_passed.append(name)
            except:
                self.pybullet_failed.append(name)

    def list_ids(self):
        """
        Returns:
            Keys of specifications.
        """
        return list(self.specs.keys()) + self.pybullet_passed

    def make(self, path, **kwargs):
        """
        Args: 
            path(string): id of object in registry
            kwargs(dict): The kwargs to pass to the object class
        Returns:
            object instance
        """
        if path in self.pybullet_failed:
            raise error.PyBulletBug("Failed to build PyBullet env with ID {}".format(path))
        if path in self.pybullet_passed:
            return pybullet_envs.make(path)

        spec = self.spec(path)
        obj = spec.make(**kwargs)
        return obj

    def register(self, id, **kwargs):
        """
        Populates the specs dict with a map from id to the object instance
        Args:
            id(string): id of object in registry
            kwargs(dict): The kwargs to pass to the object class
        """
        if id in self.specs:
            raise error.Error('Cannot re-register ID {} for {}'.format(id, self))
        if id in self.pybullet_passed + self.pybullet_failed:
            raise error.Error('ID {} for {} already exists as a PyBullet environment'.format(id, self))
        self.specs[id] = Spec(id, self.regexp, **kwargs)


# Have a global problem_registry
problem_registry = ProblemRegistry(problem_id_re)

def problem_register(id, **kwargs):
    return problem_registry.register(id, **kwargs)

def problem_spec(id):
    return problem_registry.spec(id)

def problem(id, **kwargs):
    return problem_registry.make(id, **kwargs)



