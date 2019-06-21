# test the ARMA problem class

import ctsb
import ctsb.core
from ctsb.core import Problem
from ctsb.problems.simulated.lds import LDS
import jax.numpy as np
import matplotlib.pyplot as plt


def test_lds():
    T = 100000
    n, m, d = 5, 3, 10
    problem = LDS(n, m, d)
    assert problem.T == 0

    test_output = []
    for t in range(T):
        u = np.random.normal(size=(n,))
        test_output.append(problem.step(u))

    assert problem.T == T
    plt.plot(test_output)
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    return


if __name__=="__main__":
    test_lds()