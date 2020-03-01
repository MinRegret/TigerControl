# tigercontrol init file

import tigercontrol
from tigercontrol import error
from tigercontrol.environments import environment, environment_registry
from tigercontrol.controllers import controller, controller_registry, CustomController, register_custom_controller
from tigercontrol.help import help
from tigercontrol.utils import set_key
from tigercontrol.experiments import Experiment

# initialize global random key by seeding the jax random number generator
# note: numpy is necessary because jax RNG is deterministic
import jax.random as random
GLOBAL_RANDOM_KEY = random.PRNGKey(0)
set_key()


__all__ = [
    "controller", 
    "environment", 
    "CustomController",
    "register_custom_controller", 
    "Experiment", 
    "help", 
    "set_key",
]
