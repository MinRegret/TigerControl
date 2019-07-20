from ctsb.models.optimizers.tests.test_adagrad import test_adagrad
from ctsb.models.optimizers.tests.test_sgd import test_sgd
from ctsb.models.optimizers.tests.test_ogd import test_ogd


# run all optimizers tests
def run_all_tests(steps=1000, show=False):
    print("\nrunning all optimizers tests...\n")
    test_sgd(time=1, show=show)
    test_ogd(time=1, show=show)
    test_adagrad(time=1, show=show)
    print("\nall optimizers tests passed\n")


if __name__ == "__main__":
    run_all_tests(show=True)