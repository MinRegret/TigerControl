import jax 
import jax.numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are as dare

n, m, A, B = 2, 1, np.array([[1., 1.], [0., 1.]]), np.array([[0.], [1.]])
Q, R = np.eye(N = n), np.eye(N = m)
x0 = np.zeros((n, 1))

T, H, M, lr, steps = 200, 10, 10, 0.001, 1
W = random.normal(size = (T, n, 1))
W = np.sin(np.arange(T)/(2*np.pi)).reshape((T, 1)) @ np.ones((1,2))

P = dare(A, B, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

x, loss_lqr = x0, []
for t in range(T):
    u = - K @ x
    loss_lqr.append((x.T @ Q @ x + u.T @ R @ u)[0][0])
    x = A @ x + B @ u + W[t]

def counterfact_loss(E, W):
    y, cost = np.zeros((n, 1)), 0
    for h in range(H):
        v = - K @ y + np.tensordot(E, W[h : h+M], axes = ([0, 2], [0, 1]))
        cost += (y.T @ Q @ y + v.T @ R @ v)[0][0]
        y = A @ y + B @ v + W[h+M]
    return cost

grad = jax.jit(jax.grad(counterfact_loss))
E = np.zeros((M, m, n))

x, loss_er_feed = x0, []
for t in range(T):
    u = - K @ x + (np.tensordot(E, W[t-M : t], axes = ([0, 2], [0, 1])) if t-M >= 0 else 0)
    loss_er_feed.append((x.T @ Q @ x + u.T @ R @ u)[0][0])
    x = A @ x + B @ u + W[t]
    
    if t+1-M-H >= 0:
        for _ in range(steps):            
            E = E - lr * grad(E, W[t+1-M-H: t+1])

plt.plot(loss_lqr, color = 'blue')
plt.plot(loss_er_feed, color = 'orange')
plt.show()
