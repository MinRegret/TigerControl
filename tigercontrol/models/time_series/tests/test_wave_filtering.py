import tigercontrol
import time
import jax.numpy as np
import jax.random as rand
import matplotlib.pyplot as plt
from tigercontrol.utils import generate_key

def test_wave_filtering(show_plot=False):
	# state variables
	T, n, m = 1000, 10, 10

	# model variables
	k, eta = 20, 0.00002 # update_steps is an optional variable recommended by Yi
	R_Theta = 5
	R_M = 2 * R_Theta * R_Theta * k**0.5  

	# generate random data (columns of Y)
	hidden_state_dim = 5
	h = rand.uniform(generate_key(), minval=-1, maxval=1, shape=(hidden_state_dim,)) # first hidden state h_0
	A = rand.normal(generate_key(), shape=(hidden_state_dim, hidden_state_dim))
	B = rand.normal(generate_key(), shape=(hidden_state_dim, n))
	C = rand.normal(generate_key(), shape=(m, hidden_state_dim))
	D = rand.normal(generate_key(), shape=(m, n))
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
	X = rand.normal(generate_key(), shape=(n, T))

	# generate Y according to predetermined matrices
	Y = []
	for t in range(T):
		Y.append(C.dot(h) + D.dot(X[:,t]) + rand.truncated_normal(generate_key(), 0, 0.1, shape=(m,)))
		h = A.dot(h) + B.dot(X[:,t]) + rand.truncated_normal(generate_key(), 0, 0.1, shape=(hidden_state_dim,))
	Y = np.array(Y).T # list to numpy matrix
	
	model = tigercontrol.model("WaveFiltering")
	model.initialize(n, m, k, T, eta, R_M)
	# loss = lambda y_true, y_pred: (y_true - y_pred)**2
	loss = lambda y_true, y_pred: (np.linalg.norm(y_true - y_pred))**2

	lastvalue_model = tigercontrol.model("LastValue")
	lastvalue_model.initialize()
 
	results = []
	lastvalue_results = []
	for i in range(T):
		# print(i)
		cur_y_pred = model.predict(X[:,i])
		#print(model.forecast(cur_x, 3))
		cur_y_true = Y[:,i]
		cur_loss = loss(cur_y_true, cur_y_pred)
		results.append(cur_loss)
		model.update(cur_y_true)

		lastvalue_cur_y_pred = lastvalue_model.predict(X[:,i])
		lastvalue_cur_y_true = Y[:,i]
		lastvalue_cur_loss = loss(lastvalue_cur_y_true, lastvalue_cur_y_pred)
		lastvalue_results.append(lastvalue_cur_loss)

	# print("test_wave_filtering passed")
	print(np.linalg.norm(X[:,-1]))
	print(np.linalg.norm(Y[:-1]))
	print(results[-10:-1])
	if show_plot:
		plt.plot(results)
		plt.title("WaveFiltering model on random data")
		plt.show(block=True)

		plt.plot(lastvalue_results)
		plt.title("LastValue model on random data")
		plt.show(block=True)
	
if __name__=="__main__":
	test_wave_filtering(True)

