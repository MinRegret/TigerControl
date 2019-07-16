# Problem class
# Author: John Hallman

from ctsb import error
from ctsb.problems import Problem
from ctsb.problems.registration import problem_registry


# class for implementing algorithms with enforced modularity
class CustomProblem(object):
    def __init__(self):
        pass

# verifies that a given class has the necessary minimum problem methods
def verify_valid_problem(problem_class):
    assert issubclass(problem_class, CustomProblem)
    for f in ['initialize', 'step']:
        if not callable(getattr(problem_class, f, None)):
            raise error.InvalidClass("CustomProblem is missing required method \'{}\'".format(f))

# global custom problem method
def register_custom_problem(custom_problem_class, custom_problem_id):
    assert type(custom_problem_id) == str
    verify_valid_problem(custom_problem_class)

    problem_registry.register_custom(
        id=custom_problem_id,
        custom_class=custom_problem_class,
    )

