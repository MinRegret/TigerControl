# Environment class
# Author: John Hallman

from tigercontrol import error
from tigercontrol.environments import Environment
from tigercontrol.environments.registration import environment_registry

 
class CustomEnvironment(object):
    ''' 
    Description: class for implementing algorithms with enforced modularity 
    '''

    def __init__(self):
        pass

def _verify_valid_environment(environment_class):
    ''' 
    Description: verifies that a given class has the necessary minimum environment controllers

    Args: a class
    '''
    assert issubclass(environment_class, CustomEnvironment)
    for f in ['initialize', 'step']:
        if not callable(getattr(environment_class, f, None)):
            raise error.InvalidClass("CustomEnvironment is missing required controller \'{}\'".format(f))

def register_custom_environment(custom_environment_class, custom_environment_id):
    '''
    Description: global custom environment controller
    '''
    assert type(custom_environment_id) == str
    _verify_valid_environment(custom_environment_class)

    environment_registry.register_custom(
        id=custom_environment_id,
        custom_class=custom_environment_class,
    )

