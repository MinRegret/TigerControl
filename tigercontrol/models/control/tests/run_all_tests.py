from tigercontrol.models.control.tests.test_kalman_filter import test_kalman_filter
from tigercontrol.models.control.tests.test_ode_shooting_method import test_ode_shooting_method
from tigercontrol.models.control.tests.test_lqr import test_lqr
from tigercontrol.models.control.tests.test_mppi import test_mppi

def run_all_tests(steps=1000, show=False):
    print("\nrunning all control models tests...\n")
    test_kalman_filter(steps=1000, show_plot=show)
    test_ode_shooting_method(steps=1000, show_plot=show)
    test_lqr(steps=1000, show_plot=show)
    #test_mppi(steps=1000, show_plot=show) # TODO: fix mppi
    print("\nall control models tests passed\n")

if __name__ == "__main__":
    run_all_tests(show=False)
