from tigercontrol.controllers.tests.test_kalman_filter import test_kalman_filter
from tigercontrol.controllers.tests.test_shooting import test_shooting
from tigercontrol.controllers.tests.test_lqr import test_lqr
from tigercontrol.controllers.tests.test_lqr_infinite_horizon import test_lqr_infinite_horizon
from tigercontrol.controllers.tests.test_gpc import test_gpc

def run_all_tests(steps=1000, show=False):
    print("\nrunning all controllers tests...\n")
    test_kalman_filter(steps=steps, show_plot=show)
    test_shooting(steps=steps, show_plot=show)
    test_lqr(steps=steps, show_plot=show)
    test_lqr_infinite_horizon(steps=steps, show_plot=show)
    test_gpc(steps=steps, show_plot=show)
    print("\nall controllers tests passed\n")

if __name__ == "__main__":
    run_all_tests(show=False)
