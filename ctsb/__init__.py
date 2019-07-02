# ctsb init file

import os
import sys
import warnings

import ctsb
from ctsb import error
from ctsb.problems import problem, problem_registry, problem_spec
from ctsb.models import model, CustomModel, model_registry, model_spec, register_custom_model
from ctsb.help import help
from ctsb.utils import tests, set_key

# initialize global random key by seeding the jax random number generator
# note: numpy is necessary because jax RNG is deterministic
import jax.random as random
GLOBAL_RANDOM_KEY = random.PRNGKey(0)
set_key()


__all__ = ["problem", "model", "CustomModel", "register_custom_model", "help", "set_key"]
