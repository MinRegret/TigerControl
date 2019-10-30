from tigercontrol.methods.controller.tests.run_all_tests import run_all_tests as control_method_tests
from tigercontrol.methods.time_series.tests.run_all_tests import run_all_tests as time_series_method_tests
from tigercontrol.methods.optimizers.tests.run_all_tests import run_all_tests as optimizers_tests
from tigercontrol.methods.tests.test_custom_method import test_custom_method

def run_all_tests(steps=1000, show=False):
    print("\nrunning all methods tests...\n")
    control_method_tests(steps=steps, show=show)
    time_series_method_tests(steps=steps, show=show)
    optimizers_tests(steps=steps, show=show)
    test_custom_method(steps=steps, show_plot=show)
    print("\nall methods tests passed\n")

if __name__ == "__main__":
    run_all_tests(show=False)
