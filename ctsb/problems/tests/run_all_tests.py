from test_arma import test_arma
from test_lds import test_lds
from test_lstm_output import test_lstm
from test_random import test_random
# from test_registration import test_registration
from test_rnn_output import test_rnn

def run_all_tests():
	show=True
	print("Testing simulated problems...")
	test_arma(show_plot=show)
	test_lds(show_plot=show)
	test_lstm(show_plot=show)
	test_random(show_plot=show)
	test_rnn(show_plot=show)
	print("All simulatd problems tests passed")

if __name__ == "__main__":
	run_all_tests()