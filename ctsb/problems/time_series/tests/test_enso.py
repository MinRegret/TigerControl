# test the ENSO problem class

import tigercontrol
import jax.numpy as np
import matplotlib.pyplot as plt


def test_enso(steps=1000, show_plot=False, verbose=False):
    T = steps
    problem = tigercontrol.problem("ENSO-v0")
    problem.initialize(input_signals = ['oni'])
    assert problem.T == 0

    test_output = []
    for t in range(T):
        x_t, y_t = problem.step()
        test_output.append(y_t)

    assert problem.T == T
    
    info = problem.hidden()
    if verbose:
        print(info)
    if show_plot:
        plt.plot(test_output)
        plt.title("ONI of Nino34")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    print("test_enso passed")
    return


if __name__=="__main__":
    test_enso(show_plot=True)