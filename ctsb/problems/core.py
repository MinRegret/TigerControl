# Problem class
# Author: John Hallman

from ctsb import error

# class for online control tests
class Problem(object):
    spec = None

    def initialize(self, **kwargs):
        # resets problem to time 0
        raise NotImplementedError

    def step(self, action=None):
        #Run one timestep of the problem's dynamics. 
        raise NotImplementedError

    def close(self):
        # closes the problem and returns used memory
        pass

    def help(self):
        # prints information about this class and its methods
        raise NotImplementedError

    @property
    def unwrapped(self):
        """Completely unwrap this problem.
        Returns:
            ctsb.Problem: The base non-wrapped ctsb.Problem instance
        """
        return self

    def __str__(self):
        if self.spec is None:
            return '<{} instance> call object help() method for info'.format(type(self).__name__)
        else:
            return '<{}<{}>> call object help() method for info'.format(type(self).__name__, self.spec.id)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        # propagate exception
        return False


class Wrapper(Problem):
    r"""Wraps the problem to allow a modular transformation. 

    This class is the base class for all wrappers. The subclass could override
    some methods to change the behavior of the original problem without touching the
    original code. 

    .. note::

        Don't forget to call ``super().__init__(problem)`` if the subclass overrides :meth:`__init__`.

    """
    def __init__(self, problem):
        self.problem = problem
        self.action_space = self.problem.action_space
        self.observation_space = self.problem.observation_space
        self.metadata = self.problem.metadata
        self.spec = getattr(self.problem, 'spec', None)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.problem, name)

    @classmethod
    def class_name(cls):
        return cls.__name__

    def step(self, action):
        return self.problem.step(action)

    def reset(self, **kwargs):
        return self.problem.reset(**kwargs)

    def render(self, mode='human', **kwargs):
        return self.problem.render(mode, **kwargs)

    def close(self):
        return self.problem.close()

    def seed(self, seed=None):
        return self.problem.seed(seed)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.problem)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.problem.unwrapped

