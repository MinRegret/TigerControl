# ctsb init file

import os
import sys
import warnings

import ctsb
from ctsb import error
from ctsb.problems import Problem, problem, problem_registry, problem_spec
from ctsb.models import Model, model, CustomModel, model_registry, model_spec
from ctsb.help import help
from ctsb.utils import tests, set_key

# initialize global random key by seeding the jax random number generator
# note: numpy is necessary because jax RNG is deterministic
import jax.random as random
GLOBAL_RANDOM_KEY = random.PRNGKey(0)
set_key()


__all__ = ["Problem", "problem", "Model", "model", "CustomModel", "help", "set_key"]
