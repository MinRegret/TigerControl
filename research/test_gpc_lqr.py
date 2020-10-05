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
from tigercontrol.controllers.gpc import GPC
import sys

T = 500
rep = 3
alg_name = ['LQR', 'GPC']
color_code = {'LQR': 'blue', 'GPC': 'red'}
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

class Wgen:
    def __init__(self, n, m):
        global T
        self.t = -1
        self.w = (np.sin(np.arange(T*m)/(32*np.pi)).reshape(T,m) @ np.ones((m, n))).reshape(T, n, 1)

    def next(self):
        self.t += 1
        return self.w[self.t]

def test_bpc_gpc_lqr():

    global T, rep
    T = 500 if len(sys.argv) < 2 else int(sys.argv[1])
    rep = 3 if len(sys.argv) < 3 else int(sys.argv[2])

    n, m, A, B = 2, 1, onp.array([[1., 1.], [0., 1.]]), onp.array([[0.], [1.]])

    lqr = evaluate(LQR(A, B), A, B, Wgen(n, m), quad)
    gpc = evaluate(GPC(A, B, cost_fn=quad), A, B, Wgen(n, m), quad)

    print("Average costs")
    print("LQR: ", np.mean(lqr))
    print("GPC: ", np.mean(gpc))

    print("test_gpc_lqr passed")
    return

if __name__=="__main__":
    test_bpc_gpc_lqr()
