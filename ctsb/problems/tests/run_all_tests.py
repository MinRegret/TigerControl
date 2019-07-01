from ctsb.problems.tests.test_arma import test_arma
from ctsb.problems.tests.test_lds import test_lds
from ctsb.problems.tests.test_lstm_output import test_lstm
from ctsb.problems.tests.test_random import test_random
from ctsb.problems.tests.test_rnn_output import test_rnn
from ctsb.problems.tests.test_sp500 import test_sp500
from ctsb.problems.tests.test_uci_indoor import test_uci_indoor
from ctsb.problems.tests.test_crypto import test_crypto

def run_all_tests(show=True):
    print("\nrunning all problems tests...\n")
    test_arma(show_plot=show)
    test_lds(show_plot=show)
    test_lstm(show_plot=show)
    test_random(show_plot=show)
    test_rnn(show_plot=show)
    test_sp500(show_plot=show)
    test_uci_indoor(show_plot=show)
<<<<<<< HEAD
    print("\nall problems tests passed\n")

=======
    test_crypto(show_plot=show)
    print("\nall simulated problems tests passed\n")
  
>>>>>>> 921192dfae05f70db29772c7615e1fdb169389bd
if __name__ == "__main__":
    run_all_tests()