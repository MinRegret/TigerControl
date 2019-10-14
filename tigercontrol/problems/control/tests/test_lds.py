# test the LDS problem class

import jax.numpy as np
import jax.random as random
import tigercontrol
import matplotlib.pyplot as plt
from tigercontrol.utils.random import generate_key


def test_lds(steps=1000, show_plot=False, verbose=False):
    T = steps
    n, m, d = 10, 3, 2 # state, input, and observation dimension
    problem = tigercontrol.problem("LDS-v0")

    system_params = {}
    gaussian = lambda dims: random.normal(generate_key(), shape=dims)
    for matrix, shape in {'A':(n, n), 'B':(n, m), 'C':(d, n), 'D':(d, m)}.items():
        system_params[matrix] = gaussian(shape)
    system_params_only_AC = {}
    system_params_only_AC['A'] = system_params['A']
    system_params_only_AC['C'] = system_params['C']
    x_init = gaussian((n,))
    # problem.initialize(n, m, d, partially_observable=True, noise_distribution='normal')
    # problem.initialize(n, m, d, partially_observable=True, noise_distribution=None)
    # problem.initialize(n, m, None, partially_observable=False, noise_distribution='normal')
    # problem.initialize(n, m, None, partially_observable=False, noise_distribution=None)
    # problem.initialize(n, m, None, partially_observable=False, noise_distribution=None, system_params=system_params) # should fail assert around line 92
    # problem.initialize(n, m, d, partially_observable=True, noise_distribution=None, system_params=system_params)
    # problem.initialize(n, m, d, partially_observable=True, noise_distribution=None, system_params=system_params_only_AC)
    # problem.initialize(n, m, d, partially_observable=False, noise_distribution=None, system_params=system_params_only_AC) # should fail assert around line 47
    

    # Test initialization over combinations where partially observable = True
    '''
    for nd in [None, 'normal', 'uniform']:
        for sp in [system_params, system_params_only_AC]:
            for x_0 in [None, x_init]:
                print("-----------------------------")
                print("nd:" + str(nd))
                print("sp:" + str(sp))
                print("x_0:" + str(x_0))
                problem = tigercontrol.problem("LDS-v0")
                problem.initialize(n, m, d, partially_observable=True, noise_distribution=nd, system_params=sp, initial_state=x_0)
    '''
    # Test custom noise
    custom_nd_vector = lambda x, t: (0.1 * x + np.cos(t), 0)
    custom_nd_scalar = lambda x, t: 0.1 * x + np.cos(t)
    problem = tigercontrol.problem("LDS-v0")
    problem.initialize(n, m, d, partially_observable=True, noise_distribution=custom_nd_vector)
    problem = tigercontrol.problem("LDS-v0")
    problem.initialize(n, m, d=None, partially_observable=False, noise_distribution=custom_nd_scalar)


    test_output = []
    for t in range(T):
        u = random.normal(generate_key(), shape=(m,))
        test_output.append(problem.step(u))

    info = problem.hidden()
    if verbose:
        print(info)

    assert problem.T == T
    if show_plot:
        plt.plot(test_output)
        plt.title("lds")
        plt.show(block=False)
        plt.pause(10)
        plt.close()
    print("test_lds_control passed")
    return


if __name__=="__main__":
    test_lds(show_plot=True)