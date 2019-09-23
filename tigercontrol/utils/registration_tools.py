import re
import importlib
import warnings
from difflib import get_close_matches
from tigercontrol import error

# This format is true today, but it's *not* an official spec.
# [username/](problem-name)-v(version)    problem-name is group 1, version is group 2
# id_re = re.compile(r'^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$')


def load(name):
    """
    Args:
        name(string): path of object to be registered
    Returns:
        The class of the object specified by name
    """
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class Spec(object):
    """A specification for a particular instance of the object to be registered. Used
    to register the parameters for official evaluations.

    Args:
        id (str): The official object ID
        regexp(regular expression): the format of id for objects in the registry
        entry_point (Optional[str]): The Python entrypoint of the object class (e.g. module.name:Class)
        kwargs (dict): The kwargs to pass to the object class
        tags (dict[str:any]): A set of arbitrary key-value tags on this object, including simple property=True tags

    Attributes:
        id (str): The official object ID
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
        """Instantiates an instance of the object with appropriate kwargs"""
        if self._entry_point is None:
            raise error.Error('Attempting to make deprecated {} with ID: {}. (HINT: is there a newer registered version of this object?)'.format(self, self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self._entry_point):
            obj = self._entry_point(**_kwargs)
        else:
            cls = load(self._entry_point)
            obj = cls(**_kwargs)

        # Make the object aware of which spec it came from.
        obj.spec = self
        return obj

    def get_class(self):
        """Returns the class object"""
        if self._entry_point is None:
            raise error.Error('Attempting to make deprecated {} with ID: {}. (HINT: is there a newer registered version of this object?)'.format(self, self.id))
        if callable(self._entry_point):
            obj = self._entry_point
        else:
            obj = load(self._entry_point)
        return obj

    def __repr__(self):
        return "{} Spec({})".format(str(self), self.id)

    def __str__(self):
        return "<TigerControl Spec>"

class Registry(object):
    """Register object by ID. IDs remain stable over time and are
    guaranteed to resolve to the same object dynamics (or be
    desupported). The goal is that results on a particular object
    should always be comparable, and not depend on the version of the
    code that was running.

    Args:
        specs(dict): key-value pairs of (id, corresponding object in registry)
        regexp(regular expression): format of id in registry
    """

    def __init__(self, regexp):
        self.specs = {}
        self.regexp = regexp
        self.custom = {}

    def make(self, path, **kwargs):
        """
        Args: 
            path(string): id of object in registry
            kwargs(dict): The kwargs to pass to the object class
        Returns:
            object instance
        """

        if path in self.custom:
            return self.custom[path]()

        try:
            spec = self.spec(path)
            obj = spec.make(**kwargs)
        except ModuleNotFoundError as e:
            s = "Not all dependencies have been installed.\nFull error: {}".format(path, e)
            raise error.DependencyNotInstalled(s)
        return obj

    def list_ids(self):
        """
        Returns:
            Keys of specifications.
        """
        return list(self.specs.keys()) + list(self.custom.keys())

    def all(self):
        """
        Returns:
            Values of specifications.
        """
        return list(self.specs.values())

    def spec(self, path):
        """
        Args:
            path(string): id of object in registry
        Returns:
            Instance of object in registry specified by path
        """
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
            closest = get_close_matches(id, self.list_ids(), n=1)
            if closest:
                raise error.UnregisteredObject('No registered {} with ID: {}, did you mean {}?'.format(self, id, closest[0]))
            raise error.Error('Attempted to look up malformed {} ID: {}. (All IDs must be of the form {}.)'.format(self, id.encode('utf-8'), self.regexp.pattern))

        try:
            return self.specs[id]
        except KeyError:
            # Parse the object name and check to see if it matches the non-version
            # part of a valid object (could also check the exact number here)
            name = match.group(1)
            matching_objects = [valid_name for valid_name, valid_spec in self.specs.items()
                             if name == valid_spec._name]
            if matching_objects:
                raise error.DeprecatedObject('{} with ID {} not found (valid versions include {})'.format(self, id, matching_objects))
            else:
                closest = get_close_matches(id, self.list_ids(), n=1)
                if closest:
                    raise error.UnregisteredObject('No registered {} with ID: {}, did you mean {}?'.format(self, id, closest[0]))
                raise error.UnregisteredObject('No registered {} with ID: {}'.format(self, id))

    def get_class(self, path):
        """
        Description: returns the object class corresponding to id.

        Args:
            path(string): id of object in registry
        """
        if path in self.custom:
            return self.custom[path]

        spec = self.spec(path)
        return spec.get_class()

    def register(self, id, **kwargs):
        """
        Description: Populates the specs dict with a map from id to the object instance

        Args:
            id(string): id of object in registry
            kwargs(dict): The kwargs to pass to the object class
        """
        if id in self.specs:
            raise error.Error('Cannot re-register ID {} for {}'.format(id, self))
        self.specs[id] = Spec(id, self.regexp, **kwargs)

    # register a custom model class
    def register_custom(self, id, custom_class):
        self.custom[id] = custom_class



