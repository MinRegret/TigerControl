# test the LQR infinite horizon method class

import tigercontrol
import jax.numpy as np
import matplotlib.pyplot as plt
import jax

def get_trajectory(environment, controller, T = 100):

    (environment_id, environment_params) = environment
    (controller_id, controller_params) = controller

    if(controller_params is None):
        controller_params = {}
    
    environment = tigercontrol.environment(environment_id)
    x = environment.initialize(**environment_params)
    
    controller_params['A'], controller_params['B'] = environment.A, environment.B
   
    controller = tigercontrol.controller(controller_id)
    controller.initialize(**controller_params)
    
    trajectory1, trajectory2 = [], []
    norms = []
    avg_regret = []
    cur_avg = 0

    for i in range(T):
        u = controller.get_action(x)
        x = environment.step(u)
        trajectory1.append(x)
        norms.append(np.linalg.norm(x))
        cur_avg = (i / (i + 1)) * cur_avg + (np.linalg.norm(x) + np.linalg.norm(u)) / (i + 1)
        avg_regret.append(cur_avg)

    return trajectory1, norms, avg_regret

def test_gpc(steps=500, show_plot=True):

    T = steps

    n = 1 # dimension of  the state x 
    m = 1 # control dimension
    noise_magnitude = 0.2
    noise_distribution = 'normal'

    environment_id = "LDS-v0"
    environment_params = {'n': n, 'm' : m, 'noise_magnitude' : noise_magnitude, 'noise_distribution' : noise_distribution}
    
    H, HH = 5, 30
    GPC_params = {'H' : H, 'HH' : HH}

    _, _, GPC_results = get_trajectory((environment_id, environment_params), \
                                                            ('GPC', GPC_params), T = T)
    _, _, LQR_results = get_trajectory((environment_id, environment_params), \
                                                            ('LQRInfiniteHorizon', {}), T = T)

    if(show_plot):
        plt.plot(GPC_results, label = "online GPC")
        plt.plot(LQR_results, label = "online LQR")
        plt.title("GPC and LQR on LDS")
        plt.legend()
        plt.show(block=False)
        plt.pause(30)
        plt.close()

    print("test_gpc passed")
    return

if __name__=="__main__":
    test_gpc()
