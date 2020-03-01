# test the LDS environment class

import jax.numpy as np
import jax.random as random
import tigercontrol
import matplotlib.pyplot as plt
from tigercontrol.utils.random import generate_key


def test_lds(steps=1000, show_plot=False, verbose=False):
    T = steps
    n, m, d = 10, 3, 2 # state, input, and observation dimension
    environment = tigercontrol.environment("LDS-v0")

    system_params = {}
    gaussian = lambda dims: random.normal(generate_key(), shape=dims)
    for matrix, shape in {'A':(n, n), 'B':(n, m), 'C':(d, n), 'D':(d, m)}.items():
        system_params[matrix] = gaussian(shape)
    system_params_only_AC = {}
    system_params_only_AC['A'] = system_params['A']
    system_params_only_AC['C'] = system_params['C']
    x_init = gaussian((n,))

    # Test custom noise
    custom_nd_vector = lambda x, t: (0.1 * x + np.cos(t), 0)
    custom_nd_scalar = lambda x, t: 0.1 * x + np.cos(t)
    environment = tigercontrol.environment("LDS-v0")
    environment.initialize(n, m, d, partially_observable=True, noise_distribution=custom_nd_vector)
    environment = tigercontrol.environment("LDS-v0")
    environment.initialize(n, m, d=None, partially_observable=False, noise_distribution=custom_nd_scalar) #, system_params={})


    test_output = []
    for t in range(T):
        u = random.normal(generate_key(), shape=(m,))
        test_output.append(environment.step(u))

    info = environment.hidden()
    if verbose:
        print(info)

    assert environment.T == T
    if show_plot:
        plt.plot(test_output)
        plt.title("lds")
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    print("test_lds passed")
    return


if __name__=="__main__":
    test_lds(show_plot=True)