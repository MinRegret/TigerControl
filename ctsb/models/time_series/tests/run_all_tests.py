from tigercontrol.models.time_series.tests.test_last_value import test_last_value
from tigercontrol.models.time_series.tests.test_predict_zero import test_predict_zero
from tigercontrol.models.time_series.tests.test_autoregressor import test_autoregressor
from tigercontrol.models.time_series.tests.test_rnn import test_rnn
from tigercontrol.models.time_series.tests.test_lstm import test_lstm

def run_all_tests(steps=1000, show=False):
    print("\nrunning all time series models tests...\n")
    test_last_value(steps=1000, show_plot=show)
    test_predict_zero(steps=1000, show_plot=show)
    test_autoregressor(steps=1000, show_plot=show)
    test_rnn(steps=1000, show_plot=show)
    test_lstm(steps=1000, show_plot=show)
    print("\nall time-series models tests passed\n")

if __name__ == "__main__":
    run_all_tests(show=False)
