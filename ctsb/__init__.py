# ctsb init file

import os
import sys
import warnings
import numpy as np

import ctsb
from ctsb import error
from ctsb.problems import Problem, problem, problem_registry, problem_spec
from ctsb.models import Model, model, model_registry, model_spec
from ctsb.help import help

# initialize global random key by seeding the jax random number generator
# note: numpy is necessary because jax RNG is deterministic
import jax.random as random
GLOBAL_RANDOM_KEY = random.PRNGKey(0)
ctsb.utils.set_key(int(np.random.random_integers(sys.maxsize)))


__all__ = ["Problem", "problem", "Model", "model", "help"]
