# test a generic Controller

import tigercontrol
import jax.numpy as np
import jax

def test_controller(steps=10, show_plot=True):

    T = steps

    n = 1 # dimension of  the state x 
    m = 1 # control dimension
    noise_magnitude = 0.2
    noise_distribution = 'normal'

    environment_id = "LDS-v0"
    environment_params = {'n': n, 'm' : m, 'noise_magnitude' : noise_magnitude, 'noise_distribution' : noise_distribution}

    C = np.identity(n + m) # quadratic cost
    LQR_params = {'C' : C, 'T' : T}

    LQR_results, LQR_norms, LQR_avg_results = get_trajectory((environment_id, environment_params), \
                                                            ('LQR', LQR_params), T = T)
    if(show_plot):
        plt.plot(LQR_norms, label = "LQR")
        plt.title("LQR on LDS");

    print("test_lqr passed")
    return

if __name__=="__main__":
    test_lqr()