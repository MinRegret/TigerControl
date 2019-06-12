# test the Random problem class


import ctsb
import ctsb.core
from ctsb.core import Problem
from ctsb.problems.time_series.random import Random
import numpy as np
import matplotlib.pyplot as plt


def test_random():
	random = Random()
	assert random.T == 0

	test_output = []
	for t in range(1000):
		test_output.append(random.step())

	assert random.T == 1000
	plt.plot(test_output)
	plt.show()
	plt.pause(5)
	plt.close()
	return


if __name__=="__main__":
	test_random()