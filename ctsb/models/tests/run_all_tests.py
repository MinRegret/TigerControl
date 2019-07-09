from ctsb.models.tests.test_last_value import test_last_value
from ctsb.models.tests.test_predict_zero import test_predict_zero
from ctsb.models.tests.test_autoregressor import test_autoregressor
from ctsb.models.tests.test_kalman_filter import test_kalman_filter
from ctsb.models.tests.test_shooting_method import test_shooting_method
from ctsb.models.tests.test_lqr import test_lqr
from ctsb.models.tests.test_custom_model import test_custom_model

def run_all_tests(show=True):
    print("\nrunning all models tests...\n")
    test_last_value(show_plot=show)
    test_predict_zero(show_plot=show)
    test_custom_model(show_plot=show)
    test_autoregressor(show_plot=show)
    test_kalman_filter(show_plot=show)
    test_shooting_method(show_plot=show)
    test_lqr(show_plot=show)
    print("\nall models tests passed\n")

if __name__ == "__main__":
    run_all_tests()