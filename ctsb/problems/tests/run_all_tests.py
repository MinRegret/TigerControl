from ctsb.problems.tests.test_arma import test_arma
from ctsb.problems.tests.test_lds import test_lds
from ctsb.problems.tests.test_lstm_output import test_lstm
from ctsb.problems.tests.test_random import test_random
from ctsb.problems.tests.test_rnn_output import test_rnn
from ctsb.problems.tests.test_sp500 import test_sp500
from ctsb.problems.tests.test_uci_indoor import test_uci_indoor
from ctsb.problems.tests.test_crypto import test_crypto
from ctsb.problems.tests.test_pendulum import test_pendulum

def run_all_tests(steps=1000, show=True):
    print("\nrunning all problems tests...\n")
    test_arma(steps=steps, show_plot=show)
    test_lds(steps=steps, show_plot=show)
    test_lstm(steps=steps, show_plot=show)
    test_random(steps=steps, show_plot=show)
    test_rnn(steps=steps, show_plot=show)
    test_sp500(steps=steps, show_plot=show)
    test_uci_indoor(steps=steps, show_plot=show)
    test_crypto(steps=steps, show_plot=show)
    test_pendulum(steps=steps, show_plot=show)
    print("\nall problems tests passed\n")
  
if __name__ == "__main__":
    run_all_tests()