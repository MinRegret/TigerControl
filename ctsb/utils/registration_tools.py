import re
import importlib
import warnings

from ctsb import error, logger

# This format is true today, but it's *not* an official spec.
# [username/](problem-name)-v(version)    problem-name is group 1, version is group 2
# id_re = re.compile(r'^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$')


def load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class Spec(object):
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

    def __init__(self, id, regexp, entry_point=None, kwargs=None, tags=None):
        self.id = id
        self.regexp = regexp

        if tags is None:
            tags = {}
        self.tags = tags
        
        # We may make some of these other parameters public if they're
        # useful.
        match = self.regexp.search(id)
        if not match:
            raise error.Error('Attempted to register malformed {} ID: {}. (Currently all IDs must be of the form {}.)'.format(self, id, self.regexp.pattern))
        self._name = match.group(1)
        self._entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of the problem with appropriate kwargs"""
        if self._entry_point is None:
            raise error.Error('Attempting to make deprecated {} with ID: {}. (HINT: is there a newer registered version of this object?)'.format(self, self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self._entry_point):
            obj = self._entry_point(**_kwargs)
        else:
            cls = load(self._entry_point)
            obj = cls(**_kwargs)

        # Make the problem aware of which spec it came from.
        obj.unwrapped.spec = self
        return obj

    def __repr__(self):
        return "{} Spec({})".format(self, self.id)


class Registry(object):
    """Register an problem by ID. IDs remain stable over time and are
    guaranteed to resolve to the same problem dynamics (or be
    desupported). The goal is that results on a particular problem
    should always be comparable, and not depend on the version of the
    code that was running.
    """

    def __init__(self, regexp):
        self.specs = {}
        self.regexp = regexp

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            logger.info('Making new {}: %s (%s)'.format(self), path, kwargs)
        else:
            logger.info('Making new {}: %s'.format(self), path)
        spec = self.spec(path)
        obj = spec.make(**kwargs)
        return obj

    def all(self):
        return self.specs.values()

    def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            # catch ImportError for python2.7 compatibility
            except ImportError:
                raise error.Error('A module ({}) was specified for the {} but was not found, make sure the package is installed with `pip install` before calling object'.format(mod_name, self))
        else:
            id = path

        match = self.regexp.search(id)
        if not match:
            raise error.Error('Attempted to look up malformed {} ID: {}. (Currently all IDs must be of the form {}.)'.format(self, id.encode('utf-8'), self.regexp.pattern))

        try:
            return self.specs[id]
        except KeyError:
            # Parse the problem name and check to see if it matches the non-version
            # part of a valid problem (could also check the exact number here)
            name = match.group(1)
            matching_objects = [valid_name for valid_name, valid_spec in self.specs.items()
                             if name == valid_spec._name]
            if matching_objects:
                raise error.DeprecatedObject('{} with ID {} not found (valid versions include {})'.format(self, id, matching_objects))
            else:
                raise error.UnregisteredObject('No registered {} with ID: {}'.format(self, id))

    def register(self, id, **kwargs):
        if id in self.specs:
            raise error.Error('Cannot re-register ID {} for {}'.format(id, self))
        self.specs[id] = Spec(id, self.regexp, **kwargs)


