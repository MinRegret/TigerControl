"""
Implements pseudorandomness in our program
"""

import sys
import numpy as np
import jax.random as random
import ctsb

def set_key(key=None):
    '''
    Descripton:
        Fix global random key to ensure reproducibility of results.
    Args:
        key (int): key that determines reproducible output
    '''
    if key == None:
        key = int(np.random.random_integers(sys.maxsize))
    assert type(key) == int
    ctsb.GLOBAL_RANDOM_KEY = random.PRNGKey(key)

def generate_key():
    '''
    Descripton:
        Generate random key.
    Returns:
        Random random key
    '''
    key, subkey = random.split(ctsb.GLOBAL_RANDOM_KEY)
    ctsb.GLOBAL_RANDOM_KEY = key
    return subkey

def get_global_key():
    '''
    Descripton:
        Get current global random key.
    Returns:
        Current global random key
    '''
    return ctsb.GLOBAL_RANDOM_KEY