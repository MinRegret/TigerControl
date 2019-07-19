from ctsb.models.control.tests.run_all_tests import run_all_tests as control_model_tests
from ctsb.models.time_series.tests.run_all_tests import run_all_tests as time_series_model_tests
from ctsb.models.optimizers.tests.run_all_tests import run_all_tests as optimizers_tests

def run_all_tests(steps=1000, show=False):
    print("\nrunning all models tests...\n")
    control_model_tests(steps=steps, show=show)
    time_series_model_tests(steps=steps, show=show)
    optimizers_tests(steps=steps, show=show)
    test_custom_model(steps=steps, show_plot=show)
    print("\nall models tests passed\n")

if __name__ == "__main__":
    run_all_tests(show=False)
