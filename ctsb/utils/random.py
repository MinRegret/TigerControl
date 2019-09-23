"""
Implements pseudorandomness in our program
"""

import sys
import numpy as np
import jax.random as random
import tigercontrol

def set_key(key=None):
    '''
    Descripton: Fix global random key to ensure reproducibility of results.

    Args:
        key (int): key that determines reproducible output
    '''
    if key == None:
        key = int(np.random.random_integers(sys.maxsize))
    assert type(key) == int
    tigercontrol.GLOBAL_RANDOM_KEY = random.PRNGKey(key)

def generate_key():
    '''
    Descripton: Generate random key.

    Returns:
        Random random key
    '''
    key, subkey = random.split(tigercontrol.GLOBAL_RANDOM_KEY)
    tigercontrol.GLOBAL_RANDOM_KEY = key
    return subkey

def get_global_key():
    '''
    Descripton: Get current global random key.

    Returns:
        Current global random key
    '''
    return tigercontrol.GLOBAL_RANDOM_KEY