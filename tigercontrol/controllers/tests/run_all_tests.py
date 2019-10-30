from tigercontrol.controllers.tests.test_kalman_filter import test_kalman_filter
from tigercontrol.controllers.tests.test_ode_shooting_method import test_ode_shooting_method
from tigercontrol.controllers.tests.test_lqr import test_lqr

def run_all_tests(steps=1000, show=False):
    print("\nrunning all controllers tests...\n")
    test_kalman_filter(steps=1000, show_plot=show)
    test_ode_shooting_method(steps=1000, show_plot=show)
    test_lqr(steps=1000, show_plot=show)
    print("\nall controllers tests passed\n")

if __name__ == "__main__":
    run_all_tests(show=False)
