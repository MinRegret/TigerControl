# Planner class
# Author: John Hallman

from tigercontrol import error
from tigercontrol.planners.core import Planner
from tigercontrol.planners.registration import planner_registry
from tigercontrol.utils.optimizers import Optimizer


# class for implementing algorithms with enforced modularity
class CustomPlanner(object):
    spec = None
    
    def __init__(self):
        pass
        
    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)

# verifies that a given class has the necessary minimum planner 
def verify_valid_planner(planner_class):
    assert issubclass(planner_class, CustomPlanner)
    for f in ['get_action', 'update']:
        if not callable(getattr(planner_class, f, None)):
            raise error.InvalidClass("CustomPlanner is missing required planner \'{}\'".format(f))

# global custom planner planner
def register_custom_planner(custom_planner_class, custom_planner_id):
    assert type(custom_planner_id) == str
    verify_valid_planner(custom_planner_class)

    planner_registry.register_custom(
        id=custom_planner_id,
        custom_class=custom_planner_class,
    )

