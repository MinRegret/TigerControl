import jax 
import jax.numpy as np
import tigercontrol as tc
import numpy.random as random
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are as dare
from tigercontrol.methods.control import ControlMethod

T, H, M, lr = 200, 10, 10, 0.001
n, m, A, B = 2, 1, np.array([[1., 1.], [0., 1.]]), np.array([[0.], [1.]])
Q, R = np.eye(N = n), np.eye(N = m)
x0 = np.zeros((n, 1))

Wproc = lambda n, x, u, w, t: random.normal(size = (n, 1))
Wproc = lambda n, x, u, w, t: np.sin(t/(2*np.pi))*np.ones((2, 1))

env = tc.problem('LDS-v0')

class LQR(ControlMethod):
    def __init__(self, A, B, Q, R):
        P = dare(A, B, Q, R)
        self.K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

    def plan(self, x):
        return -self.K @ x

x = env.initialize(n, m, noise_distribution = Wproc, system_params = {'A': A, 'B': B}, initial_state = x0)
lqr, loss_lqr = LQR(A, B, Q, R), []
for t in range(T):
    u = lqr.plan(x)
    loss_lqr.append(np.linalg.norm(x)**2 + np.linalg.norm(u)**2)
    x = env.step(u)

class GPC(ControlMethod):
    def __init__(self, A, B, Q, R, M, H, lr):
        n, m = B.shape
        self.M, self.H = M, H
        self.A, self.B, self.x, self.u = A, B, np.zeros((n, 1)), np.zeros((m, 1))
        self.K, self.E = LQR(A, B, Q, R).K, np.zeros((M, m, n))
        self.W, self.lr, self.t = np.zeros((H + M, n, 1)), lr, 0

        def counterfact_loss(E, W):
            y, cost = np.zeros((n, 1)), 0
            for h in range(H):
                v = -self.K @ y + np.tensordot(E, W[h : h + M], axes = ([0, 2], [0, 1]))
                cost = (y.T @ Q @ y + v.T @ R @ v)[0][0]
                y = A @ y + B @ v + W[h + M]
            return cost
        self.grad = jax.jit(jax.grad(counterfact_loss))

    def plan(self, x):
        self.W = jax.ops.index_update(self.W, 0, x - self.A @ self.x - self.B @ self. u)
        print(self.W[0])
        self.W = np.roll(self.W, -1, axis = 0)
        if self.t > self.H + self.M:
            self.E = self.E - self.lr * self.grad(self.E, self.W)
        self.x, self.u, self.t = x, -self.K @ x + np.tensordot(self.E, self.W[-self.M:], axes = ([0, 2], [0, 1])), self.t + 1
        return self.u

x = env.initialize(n, m, noise_distribution = Wproc, system_params = {'A': A, 'B': B}, initial_state = x0)
gpc, loss_gpc = GPC(A, B, Q, R, M, H, lr), []
for t in range(T):
    u = gpc.plan(x)
    loss_gpc.append(np.linalg.norm(x)**2 + np.linalg.norm(u)**2)
    x = env.step(u)

plt.plot(loss_lqr, color = 'blue')
plt.plot(loss_gpc, color = 'orange')
plt.show()
