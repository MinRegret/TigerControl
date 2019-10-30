# tigercontrol init file

import os
import sys
import warnings

import tigercontrol
from tigercontrol import error
from tigercontrol.environments import environment, CustomEnvironment, environment_registry, register_custom_environment
from tigercontrol.methods import method, CustomMethod, method_registry, register_custom_method
from tigercontrol.methods.optimizers import losses
from tigercontrol.help import help
from tigercontrol.utils import set_key
from tigercontrol.experiments import Experiment

# initialize global random key by seeding the jax random number generator
# note: numpy is necessary because jax RNG is deterministic
import jax.random as random
GLOBAL_RANDOM_KEY = random.PRNGKey(0)
set_key()


__all__ = [
	"environment", 
	"method", 
	"CustomMethod", 
	"Experiment", 
	"register_custom_method", 
	"register_custom_environment", 
	"help", 
	"set_key"
]
