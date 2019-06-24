from ctsb.problems.tests.test_arma import test_arma
from ctsb.problems.tests.test_lds import test_lds
from ctsb.problems.tests.test_lstm_output import test_lstm
from ctsb.problems.tests.test_random import test_random
from ctsb.problems.tests.test_rnn_output import test_rnn

def run_all_tests():
	show=True
	print("Testing simulated problems...")
	test_arma(show_plot=show)
	test_lds(show_plot=show)
	test_lstm(show_plot=show)
	test_random(show_plot=show)
	test_rnn(show_plot=show)
	print("All simulated problems tests passed")

if __name__ == "__main__":
	run_all_tests()