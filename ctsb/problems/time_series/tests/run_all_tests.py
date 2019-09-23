from tigercontrol.problems.time_series.tests.test_arma import test_arma
from tigercontrol.problems.time_series.tests.test_random import test_random
from tigercontrol.problems.time_series.tests.test_sp500 import test_sp500
from tigercontrol.problems.time_series.tests.test_uci_indoor import test_uci_indoor
from tigercontrol.problems.time_series.tests.test_crypto import test_crypto
from tigercontrol.problems.time_series.tests.test_enso import test_enso

def run_all_tests(steps=1000, show=False):
    print("\nrunning all time-series problems tests...\n")
    test_arma(steps=steps, show_plot=show)
    test_random(steps=steps, show_plot=show)
    test_sp500(steps=steps, show_plot=show)
    test_uci_indoor(steps=steps, show_plot=show)
    test_crypto(steps=steps, show_plot=show)
    test_enso(steps=steps, show_plot=show)
    print("\nall time-series problems tests passed\n")
  
if __name__ == "__main__":
    run_all_tests(show=True)