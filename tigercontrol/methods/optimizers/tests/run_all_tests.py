from tigercontrol.methods.optimizers.tests.test_adagrad import test_adagrad
from tigercontrol.methods.optimizers.tests.test_sgd import test_sgd
from tigercontrol.methods.optimizers.tests.test_ogd import test_ogd
from tigercontrol.methods.optimizers.tests.test_adam import test_adam
from tigercontrol.methods.optimizers.tests.test_ons import test_ons


# run all optimizers tests
def run_all_tests(steps=1000, show=False):
    print("\nrunning all optimizers tests...\n")
    test_sgd(show=show)
    test_ogd(show=show)
    test_adagrad(show=show)
    test_adam(show=show)
    test_ons(show=show)
    print("\nall optimizers tests passed\n")


if __name__ == "__main__":
    run_all_tests(show=False)
