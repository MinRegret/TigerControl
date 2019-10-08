from tigercontrol.methods.time_series.tests.test_last_value import test_last_value
from tigercontrol.methods.time_series.tests.test_predict_zero import test_predict_zero
from tigercontrol.methods.time_series.tests.test_autoregressor import test_autoregressor
from tigercontrol.methods.time_series.tests.test_rnn import test_rnn
from tigercontrol.methods.time_series.tests.test_lstm import test_lstm

def run_all_tests(steps=1000, show=False):
    print("\nrunning all time series methods tests...\n")
    test_last_value(steps=1000, show_plot=show)
    test_predict_zero(steps=1000, show_plot=show)
    test_autoregressor(steps=1000, show_plot=show)
    test_rnn(steps=1000, show_plot=show)
    test_lstm(steps=1000, show_plot=show)
    print("\nall time-series methods tests passed\n")

if __name__ == "__main__":
    run_all_tests(show=False)
