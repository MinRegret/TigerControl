from ctsb.problems.time_series.tests.test_arma import test_arma
from ctsb.problems.time_series.tests.test_random import test_random
from ctsb.problems.time_series.tests.test_sp500 import test_sp500
from ctsb.problems.time_series.tests.test_uci_indoor import test_uci_indoor
from ctsb.problems.time_series.tests.test_crypto import test_crypto

def run_all_tests(steps=1000, show=True):
    print("\nrunning all time-series problems tests...\n")
    test_arma(steps=steps, show_plot=show)
    test_random(steps=steps, show_plot=show)
    test_sp500(steps=steps, show_plot=show)
    test_uci_indoor(steps=steps, show_plot=show)
    test_crypto(steps=steps, show_plot=show)
    print("\nall time-series problems tests passed\n")
  
if __name__ == "__main__":
    run_all_tests()