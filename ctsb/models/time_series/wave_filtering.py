# Wave filtering from paper "Learning Linear Dynamical Systems via Spectral Filtering", 2017
# John Hallman

import time
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# state variables
T, n, m = 5000, 10, 10

# model variables
k, eta = 20, 0.00002 # update_steps is an optional variable recommended by Yi
R_Theta = 5
R_M = 2 * R_Theta * R_Theta * k**0.5 


# compute and return top k eigenpairs of a TxT Hankel matrix
def eigen_pairs(T, k):
	v = np.fromfunction(lambda i: 1.0 / ((i+2)**3 - (i+2)), (2 * T - 1,))
	Z = 2 * la.hankel(v[:T], v[T-1:])
	eigen_values, eigen_vectors = np.linalg.eigh(Z)
	return eigen_values[-k:], eigen_vectors[:,-k:]


# Wave filtering with squared loss, note T > 4 * k is recommended.
def wave_filter(T, X, Y, k, eta, R_M, k_values, k_vectors):
	assert(k_vectors.shape[0] == T)
	n = X.shape[0]
	m = Y.shape[0]
	if (4 * k > T):
		raise Exception("Model parameter k must be less than T/4")

	# initialize M_1
	k_prime = n * k + 2 * n + m
	M = 2 * np.random.rand(m, k_prime) - 1

	# iterate over data points in Y
	loss = []
	for t in range(X.shape[1]):
		if (t == 0): # t = 0 results in an excessively complicated corner case otherwise
			X_sim = np.append(np.zeros(n * k + n), np.append(X[:,0], np.zeros(m)))
		else:
			eigen_diag = np.diag(k_values**0.25)
			if (t <= T):
				X_sim_pre = X[:,0:t-1].dot(np.flipud(k_vectors[0:t-1,:])).dot(eigen_diag)
			else:
				X_sim_pre = X[:,t-T-1:t-1].dot(np.flipud(k_vectors)).dot(eigen_diag)
			x_y_cols = np.append(np.append(X[:,t-1], X[:,t]), Y[:,t-1])
			X_sim = np.append(X_sim_pre.T.flatten(), x_y_cols)
		y_hat = M.dot(X_sim)
		y_delta = Y[:,t] - y_hat
		loss.append(y_delta.dot(y_delta))
		M = M - 2 * eta * np.outer(y_delta, X_sim) # changed from +2 to -2
		if (np.linalg.norm(M) > R_M):
			M = M * (R_M / np.linalg.norm(M))
	return loss


# runs tests (outdated)
def runtests():
	# generate random data (columns of Y)
	hidden_state_dim = 5
	h = np.random.uniform(-1, 1, hidden_state_dim) # first hidden state h_0
	A = np.random.normal(size=(hidden_state_dim, hidden_state_dim))
	B = np.random.normal(size=(hidden_state_dim, n))
	C = np.random.normal(size=(m, hidden_state_dim))
	D = np.random.normal(size=(m, n))
	A = (A + A.T) / 2 # make A symmetric
	A = 0.99 * A / np.linalg.norm(A, ord=2)
	if (np.linalg.norm(B) > R_Theta):
		B = B * (R_Theta / np.linalg.norm(B))
	if (np.linalg.norm(C) > R_Theta):
		C = C * (R_Theta / np.linalg.norm(C))
	if (np.linalg.norm(D) > R_Theta):
		D = D * (R_Theta / np.linalg.norm(D))
	if (np.linalg.norm(h) > R_Theta):
		h = h * (R_Theta / np.linalg.norm(h))


	# input vectors are random data
	X = np.random.normal(size=(n, T))

	# generate Y according to predetermined matrices
	Y = []
	for t in range(T):
		Y.append(C.dot(h) + D.dot(X[:,t]) + np.random.normal(0, 0.1, m))
		h = A.dot(h) + B.dot(X[:,t]) + np.random.normal(0, 0.1, hidden_state_dim)
	Y = np.array(Y).T # list to numpy matrix

	# compute eigenpairs
	vals, vecs = eigen_pairs(T, k)

	# run ARMA_ONS on random data
	wave_filter(X, Y, k, eta, R_M, vals, vecs)