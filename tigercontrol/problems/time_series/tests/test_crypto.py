# test the crypto problem class

import tigercontrol
import jax.numpy as np
import matplotlib.pyplot as plt


def test_crypto(steps=1000, show_plot=False, verbose=False):
    T = steps
    problem = tigercontrol.problem("Crypto-v0")
    problem.initialize()
    assert problem.T == 0

    test_output = []
    for t in range(T):
        test_output.append(problem.step())

    assert problem.T == T
    
    info = problem.hidden()
    if verbose:
        print(info)
    if show_plot:
        plt.plot(test_output)
        plt.title("Crypto")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    print("test_crypto passed")
    return


if __name__=="__main__":
    test_crypto(show_plot=True)