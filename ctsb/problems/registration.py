import re
import importlib
import warnings

from ctsb import error, logger

# This format is true today, but it's *not* an official spec.
# [username/](problem-name)-v(version)    problem-name is group 1, version is group 2
problem_id_re = re.compile(r'^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$')


def load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class ProblemSpec(object):
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

    def __init__(self, id, entry_point=None, kwargs=None, nondeterministic=False, tags=None, max_episode_steps=None):
        self.id = id
        # Problemironment properties
        self.nondeterministic = nondeterministic

        if tags is None:
            tags = {}
        self.tags = tags

        tags['wrapper_config.TimeLimit.max_episode_steps'] = max_episode_steps
        
        self.max_episode_steps = max_episode_steps

        # We may make some of these other parameters public if they're
        # useful.
        match = problem_id_re.search(id)
        if not match:
            raise error.Error('Attempted to register malformed problem ID: {}. (Currently all IDs must be of the form {}.)'.format(id, problem_id_re.pattern))
        self._problem_name = match.group(1)
        self._entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of the problem with appropriate kwargs"""
        if self._entry_point is None:
            raise error.Error('Attempting to make deprecated problem {}. (HINT: is there a newer registered version of this problem?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self._entry_point):
            problem = self._entry_point(**_kwargs)
        else:
            cls = load(self._entry_point)
            problem = cls(**_kwargs)

        # Make the problem aware of which spec it came from.
        problem.unwrapped.spec = self

        return problem

    def __repr__(self):
        return "ProblemSpec({})".format(self.id)


class ProblemRegistry(object):
    """Register an problem by ID. IDs remain stable over time and are
    guaranteed to resolve to the same problem dynamics (or be
    desupported). The goal is that results on a particular problem
    should always be comparable, and not depend on the version of the
    code that was running.
    """

    def __init__(self):
        self.problem_specs = {}

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            logger.info('Making new problem: %s (%s)', path, kwargs)
        else:
            logger.info('Making new problem: %s', path)
        spec = self.spec(path)
        problem = spec.make(**kwargs)
        # We used to have people override _reset/_step rather than
        # reset/step. Set _ctsb_disable_underscore_compat = True on
        # your problem if you use these methods and don't want
        # compatibility code to be invoked.
        if hasattr(problem, "_reset") and hasattr(problem, "_step") and not getattr(problem, "_ctsb_disable_underscore_compat", False):
            patch_deprecated_methods(problem)
        if (problem.spec.max_episode_steps is not None) and not spec.tags.get('vnc'):
            from ctsb.wrappers.time_limit import TimeLimit
            problem = TimeLimit(problem, max_episode_steps=problem.spec.max_episode_steps)
        return problem

    def all(self):
        return self.problem_specs.values()

    def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            # catch ImportError for python2.7 compatibility
            except ImportError:
                raise error.Error('A module ({}) was specified for the problem but was not found, make sure the package is installed with `pip install` before calling `ctsb.make()`'.format(mod_name))
        else:
            id = path

        match = problem_id_re.search(id)
        if not match:
            raise error.Error('Attempted to look up malformed problem ID: {}. (Currently all IDs must be of the form {}.)'.format(id.encode('utf-8'), problem_id_re.pattern))

        try:
            return self.problem_specs[id]
        except KeyError:
            # Parse the problem name and check to see if it matches the non-version
            # part of a valid problem (could also check the exact number here)
            problem_name = match.group(1)
            matching_problems = [valid_problem_name for valid_problem_name, valid_problem_spec in self.problem_specs.items()
                             if problem_name == valid_problem_spec._problem_name]
            if matching_problems:
                raise error.DeprecatedProblem('problem {} not found (valid versions include {})'.format(id, matching_problems))
            else:
                raise error.UnregisteredProblem('No registered problem with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.problem_specs:
            raise error.Error('Cannot re-register id: {}'.format(id))
        self.problem_specs[id] = ProblemSpec(id, **kwargs)

# Have a global registry
registry = ProblemRegistry()

def register(id, **kwargs):
    return registry.register(id, **kwargs)

def make(id, **kwargs):
    return registry.make(id, **kwargs)

def spec(id):
    return registry.spec(id)

warn_once = True



