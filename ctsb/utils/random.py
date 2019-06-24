"""
Implements pseudorandomness in our program
"""

import sys
import numpy as np
import jax.random as random
import ctsb

def set_key(key=None):
    if key == None:
        key = int(np.random.random_integers(sys.maxsize))
    assert type(key) == int
    ctsb.GLOBAL_RANDOM_KEY = random.PRNGKey(key)

def generate_key():
    key, subkey = random.split(ctsb.GLOBAL_RANDOM_KEY)
    ctsb.GLOBAL_RANDOM_KEY = key
    return subkey

def get_global_key():
    return ctsb.GLOBAL_RANDOM_KEY