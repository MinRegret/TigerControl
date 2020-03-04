# test the LQR method class

import tigercontrol
import jax.numpy as np
import matplotlib.pyplot as plt
import jax

def get_trajectory(environment, controller, T = 100):

    (environment_id, environment_params) = environment
    (controller_id, controller_params) = controller
    
    environment = tigercontrol.environment(environment_id)
    x = environment.initialize(**environment_params)
    
    controller_params['A'], controller_params['B'] = environment.A, environment.B
    
    controller = tigercontrol.controller(controller_id)
    controller.initialize(**controller_params)
    
    trajectory = []
    norms = []
    avg_regret = []
    cur_avg = 0
    
    u = controller.plan(x, T)

    for i in range(T):
        x = environment.step(u[i])
        trajectory.append(x)
        norms.append(np.linalg.norm(x))
        cur_avg = (i / (i + 1)) * cur_avg + (np.linalg.norm(x) + np.linalg.norm(u[i])) / (i + 1)
        avg_regret.append(cur_avg)
            
    return trajectory, norms, avg_regret

def test_lqr(steps=10, show_plot=True):

    T = steps

    n = 1 # dimension of  the state x 
    m = 1 # control dimension
    noise_magnitude = 0.2
    noise_distribution = 'normal'

    environment_id = "LDS"
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
