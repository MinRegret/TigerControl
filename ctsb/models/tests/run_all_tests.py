from ctsb.models.tests.test_last_value import test_last_value
from ctsb.models.tests.test_predict_zero import test_predict_zero
from ctsb.models.tests.test_linear import test_linear

def run_all_tests(show=True):
    print("\nrunning all models tests...\n")
	test_last_value(show_plot=show)
	test_predict_zero(show_plot=show)
	test_linear(show_plot=show)
	test_kalman_filter(show_plot=show)
    print("\nall models tests passed\n")

if __name__ == "__main__":
	run_all_tests()