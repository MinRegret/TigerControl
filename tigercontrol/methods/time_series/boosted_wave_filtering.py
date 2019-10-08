# Online gradient boosting from paper "Online Gradient Boosting", 2015 applied to
# Wave filtering from paper "Learning Linear Dynamical Systems via Spectral Filtering", 2017
# John Hallman

import numpy as np
import scipy.linalg as sci_la

# online wave filtering algorithm object
class Wave_filter(object):
	def __init__(self, X, Y, eigenvalues, eigenvectors, R_M, eta):
		self.t = 0
		self.X = X
		self.Y = Y
		self.R_M = R_M
		self.eta = eta
		self.vals = eigenvalues
		self.vecs = eigenvectors
		n, m, k = X.shape[0], Y.shape[0], eigenvalues.shape[0]
		self.X_sim = np.append(np.zeros(n * k + n), np.append(X[:,0], np.zeros(m)))
		self.M = np.random.normal(0, 1, size=(m, n*k + 2*n + m))

	def predict(self, t):
		assert(t == self.t)
		M, X, Y = self.M, self.X, self.Y
		if (t > 0):
			X_sim_pre = X[:,0:t-1].dot(np.flipud(self.vecs[0:t-1,:])).dot(np.diag(self.vals))
			x_y_cols = np.append(np.append(X[:,t-1], X[:,t]), Y[:,t-1])
			self.X_sim = np.append(X_sim_pre.T.flatten(), x_y_cols)
		return self.M.dot(self.X_sim)

	def update(self, grad):
		self.M = self.M - self.eta * np.outer(grad, self.X_sim.T)
		if (np.linalg.norm(self.M) > self.R_M):
			self.M = self.M * (self.R_M / np.linalg.norm(self.M))
		self.t += 1


# OGB with convex hull of F for wave filtering, square loss
# N is number of copies of algorithm A
# X is data set, Y is correct output, L_D is Lipschitz constant
# A_eta, k, R_M are parameters for algorithm A
def OGB_CH_wave_filtering(N, X, Y, L_D, A_eta, k, R_M, vals, vecs, projection):
	T = X.shape[1]
	D = 0.5 * L_D
	if projection:
		project = lambda v, b: v * (b / np.linalg.norm(v)) if (np.linalg.norm(v) > b) else v
	else:
		project = lambda v, b: v
	A = [Wave_filter(X, Y, vals, vecs, R_M, A_eta) for i in range(N)]
	eta = [2.0 / (i+2) for i in range(N)]

	loss = []
	for t in range(T):
		y = np.zeros((Y.shape[0], N+1))
		for i in range(N):
			y[:,i+1] = project((1 - eta[i]) * y[:,i] + eta[i] * A[i].predict(t), D)
		y_predict = y[:,N]
		loss.append(0.5 * (y_predict - Y[:,t]).dot(y_predict - Y[:,t]))
		for i in range(N):
			A[i].update((y[:,i] - Y[:,t]) / L_D)
	return loss


# OGB using span(F) for wave filtering, square loss
# N is number of copies of algorithm A
# X is data set, Y is correct output, L_B is Lipschitz constant
# eta is learning rate for the boosting algorithm, B is update parameter
# A_eta, k, R_M are parameters for algorithm A
def OGB_span_wave_filtering(N, X, Y, L_B, eta, A_eta, k, R_M, vals, vecs):
	assert(eta >= 1.0 / N and eta <= 1)
	T = X.shape[1]
	B = 0.5 * L_B
	project = lambda v, b: v * (b / np.linalg.norm(v)) if (np.linalg.norm(v) > b) else v
	A = [Wave_filter(X, Y, vals, vecs, R_M, A_eta) for i in range(N)]
	sigma = np.zeros(N)

	loss = []
	for t in range(T):
		y = np.zeros((Y.shape[0], N+1))
		for i in range(N):
			y[:,i+1] = project((1 - sigma[i] * eta) * y[:,i] + eta * A[i].predict(t), 😎
		y_predict = y[:,N]
		loss.append(0.5 * (y_predict - Y[:,t]).dot(y_predict - Y[:,t]))
		for i in range(N):
			A[i].update((y[:,i] - Y[:,t]) / L_B)
			sigma[i] = max(0, min(1, sigma[i] + (y[:,i] - Y[:,t]).dot(y[:,i]) / (L_B * B * (t+1)**0.5)))
	return loss