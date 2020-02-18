# test the LQR infinite horizon method class

import tigercontrol
import jax.numpy as np
import matplotlib.pyplot as plt
import jax
import pandas as pd
import numpy as onp
import numpy.random as random
import seaborn as sns
from tigercontrol.controllers.lqr import LQR
from tigercontrol.controllers.bpc import BPC
from tigercontrol.controllers.gpc_v1 import GPC_v1
from tigercontrol.controllers.gpc_v2 import GPC_v2

T = 500
alg_name = ['LQR', 'GPC-v1', 'GPC-v2', 'BPC']
color_code = {'LQR': 'blue', 'GPC-v1': 'red', 'GPC-v2': 'orange', 'BPC': 'purple'}
quad = lambda x, u: np.sum(x.T @ x + u.T @ u)

def evaluate(controller, A, B, Wgen, cost_fn):
    global T
    x, loss = np.zeros(Wgen.w[0].shape), [0. for _ in range(T)]
    for t in range(T):
        u = controller.get_action(x)
        loss[t] = cost_fn(x, u)
        controller.update(loss[t])
        x = A @ x + B @ u + Wgen.next()

    loss = np.array(loss, dtype=np.float32)
    return loss, np.cumsum(loss)/np.arange(1, T+1)


def to_dataframe(alg, loss):
    inst_loss, avg_loss = loss
    return pd.DataFrame(data = {'Algorithm': alg, 'Time': np.arange(T, dtype=np.float32),
                                'Instantaneous Cost': inst_loss, 'Average Cost': avg_loss})

def benchmark(A, B, Wgen, cost_fn = quad):

    loss_lqr = evaluate(LQR(A, B), A, B, Wgen, cost_fn)
    loss_gpc_1 = evaluate(GPC_v1(A, B, cost_fn=cost_fn), A, B, Wgen, cost_fn)
    loss_gpc_2 = evaluate(GPC_v2(A, B, cost_fn=cost_fn), A, B, Wgen, cost_fn)
    loss_bpc = evaluate(BPC(A, B), A, B, Wgen, cost_fn)

    return loss_lqr, loss_gpc_1, loss_gpc_2, loss_bpc

def repeat_benchmark(A, B, Wgen, rep = 3, cost_fn = quad):
    all_data = []
    for r in range(rep):
        loss = benchmark(A, B, Wgen, cost_fn)
        data = pd.concat(list(map(lambda x: to_dataframe(*x), list(zip(alg_name, loss)))))
        all_data.append(data)
    all_data = pd.concat(all_data)
    return all_data[all_data['Instantaneous Cost'].notnull()]

def plot(title, data, scale = 'linear'):
    fig, axs = plt.subplots(ncols=2, figsize=(15,4))
    axs[0].set_yscale(scale)
    sns.lineplot(x = 'Time', y = 'Instantaneous Cost', hue = 'Algorithm', 
                 data = data, ax = axs[0], ci = 'sd', palette = color_code).set_title(title)
    axs[1].set_yscale(scale)
    sns.lineplot(x = 'Time', y = 'Average Cost', hue = 'Algorithm', 
                 data = data, ax = axs[1], ci = 'sd', palette = color_code).set_title(title)

class Wgen:
    def __init__(self, n, m):
        global T
        self.t = 0
        self.w = (np.sin(np.arange(T*m)/(32*np.pi)).reshape(T,m) @ np.ones((m, n))).reshape(T, n, 1)

    def next(self):
        self.t += 1
        return self.w[self.t]

def test_pc_vs_lqr(T=500, rep = 3, show_plot=True):

    T = T

    n, m, A, B = 2, 1, onp.array([[1., 1.], [0., 1.]]), onp.array([[0.], [1.]])
    Q, R = onp.array([[1., 0.], [0., 1.]]), onp.array([[1.]])

    quad_cost = lambda x, u: (x.T @ Q @ x + u.T @ R @ u)[0][0]

    data = repeat_benchmark(A, B, Wgen(n, m), rep = 1)
    plot('Sinusoidal Perturbations', data)

    print("test_gpc passed")
    return

if __name__=="__main__":
    test_pc_vs_lqr()
