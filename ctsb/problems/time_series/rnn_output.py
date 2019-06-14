"""
Produces an ARMA sequence based on given hyperparameters p,q.
"""

import ctsb
import numpy as np
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Input, Dense, SimpleRNN


# class for online control tests
class RNN_Output(ctsb.Problem):

	# n: input dimension. m: observation dimension. l: length of RNN memory. h: RNN hidden dimension
	def __init__(self, n, m, l, h):
		self.T = 0
		self.n, self.m, self.l, self.h = n, m, l, h

		hidden = SimpleRNN(h, input_shape=(l,n))
		output = Dense(m)
		model = Sequential()
		model.add(hidden)
		model.add(output)
		model.compile(loss='mse', optimizer='sgd')
		hidden_model = Sequential()
		hidden_model.add(hidden)
		hidden_model.compile(loss='mse', optimizer='sgd')

		self.model = model
		self.hidden_model = hidden_model
		self.x = np.zeros(shape=(l,n))

	def seed(self, seed=None):
		#self.np_random, seed = seeding.np_random()
		return 1 #[seed]

	def step(self, x):
		assert x.shape == (self.n,)
		self.T += 1
		self.x[1:,:] = self.x[:-1,:]
		self.x[0,:] = x

		y = self.model.predict(self.x.reshape(1, self.n, self.l))[0]
		return y

	def reset(self):
		self.T = 0
		self.x = np.zeros(shape=(self.n,self.l))

	def hidden(self):
		return self.hidden_model.predict(self.x.reshape(1, self.n, self.l))[0]

	def close(self):
		pass


