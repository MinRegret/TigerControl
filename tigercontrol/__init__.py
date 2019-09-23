# tigercontrol init file

import os
import sys
import warnings

import tigercontrol
from tigercontrol import error
from tigercontrol.problems import problem, CustomProblem, problem_registry, register_custom_problem
from tigercontrol.models import model, CustomModel, model_registry, register_custom_model
from tigercontrol.models.optimizers import losses
from tigercontrol.help import help
from tigercontrol.utils import set_key
from tigercontrol.experiments import Experiment

# initialize global random key by seeding the jax random number generator
# note: numpy is necessary because jax RNG is deterministic
import jax.random as random
GLOBAL_RANDOM_KEY = random.PRNGKey(0)
set_key()


__all__ = [
	"problem", 
	"model", 
	"CustomModel", 
	"Experiment", 
	"register_custom_model", 
	"register_custom_problem", 
	"help", 
	"set_key"
]
