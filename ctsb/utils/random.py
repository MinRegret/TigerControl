"""
Implements pseudorandomness in our program
"""

import jax.random as random
import ctsb

def set_key(key):
    assert type(key) == int
    ctsb.GLOBAL_RANDOM_KEY = random.PRNGKey(key)

def generate_key():
    key, subkey = random.split(ctsb.GLOBAL_RANDOM_KEY)
    ctsb.GLOBAL_RANDOM_KEY = key
    return subkey

def get_global_key():
    return ctsb.GLOBAL_RANDOM_KEY