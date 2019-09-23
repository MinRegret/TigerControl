# Problem class
# Author: John Hallman

from tigercontrol import error
from tigercontrol.problems import Problem
from tigercontrol.problems.registration import problem_registry

 
class CustomProblem(object):
    ''' 
    Description: class for implementing algorithms with enforced modularity 
    '''

    def __init__(self):
        pass

def _verify_valid_problem(problem_class):
    ''' 
    Description: verifies that a given class has the necessary minimum problem methods

    Args: a class
    '''
    assert issubclass(problem_class, CustomProblem)
    for f in ['initialize', 'step']:
        if not callable(getattr(problem_class, f, None)):
            raise error.InvalidClass("CustomProblem is missing required method \'{}\'".format(f))

def register_custom_problem(custom_problem_class, custom_problem_id):
    '''
    Description: global custom problem method
    '''
    assert type(custom_problem_id) == str
    _verify_valid_problem(custom_problem_class)

    problem_registry.register_custom(
        id=custom_problem_id,
        custom_class=custom_problem_class,
    )

