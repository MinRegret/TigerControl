# test random number generator

import sys
import numpy as np
import jax.random as random
from tigercontrol.utils.random import set_key, generate_key, get_global_key

# test jax random number generator seeding implementation
def test_random(show=False):
    set_key(5)
    a1 = get_global_key()
    r1 = generate_key()
    set_key(5)
    a2 = get_global_key()
    r2 = generate_key()
    assert str(a1) == str(a2)
    assert str(r1) == str(r2)
    print("test_random passed")

if __name__ == "__main__":
    test_random()
