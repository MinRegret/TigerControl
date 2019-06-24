# -*- coding: utf-8 -*-
import ctsb
from ctsb import error, problems
from ctsb.problems import registration
from ctsb.problems.simulated import random
from test_random import test_random

class ArgumentProblem(ctsb.Problem):
    def __init__(self, arg1, arg2, arg3):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3

ctsb.register(
    id='test.ArgumentProblem-v0',
    entry_point='ctsb.problems.tests.test_registration:ArgumentProblem',
    kwargs={
        'arg1': 'arg1',
        'arg2': 'arg2',
    }
)

def test_make():
    problem = problems.make('Random-v0')
    assert problem.spec.id == 'Random-v0'
    #test_random()
    #assert isinstance(problem.unwrapped, cartpole.CartPoleProblem)
    return

def test_make_with_kwargs():
    problem = problems.make('test.ArgumentProblem-v0', arg2='override_arg2', arg3='override_arg3')
    assert problem.spec.id == 'test.ArgumentProblem-v0'
    assert isinstance(problem.unwrapped, ArgumentProblem)
    assert problem.arg1 == 'arg1'
    assert problem.arg2 == 'override_arg2'
    assert problem.arg3 == 'override_arg3'

def test_make_deprecated():
    try:
        problems.make('Random-v0')
    except error.Error:
        pass
    else:
        assert False

def test_spec():
    spec = problems.spec('Random-v0')
    assert spec.id == 'Random-v0'

def test_missing_lookup():
    registry = registration.ProblemRegistry()
    registry.register(id='Test-v0', entry_point=None)
    registry.register(id='Test-v15', entry_point=None)
    registry.register(id='Test-v9', entry_point=None)
    registry.register(id='Other-v100', entry_point=None)
    try:
        registry.spec('Test-v1')  # must match an problem name but not the version above
    except error.DeprecatedProblem:
        pass
    else:
        assert False

    try:
        registry.spec('Unknown-v1')
    except error.UnregisteredProblem:
        pass
    else:
        assert False

def test_malformed_lookup():
    registry = registration.ProblemRegistry()
    try:
        registry.spec('Random-v0')
    except error.Error as e:
        assert 'malformed problemironment ID' in '{}'.format(e), 'Unexpected message: {}'.format(e)
    else:
        assert False

def test_registration():
    print("\n--- Testing registration ---\n")

    print("test_make")
    test_make()

    print("test_make_with_kwargs")
    test_make_with_kwargs()

    print("test_make_deprecated")
    test_make_deprecated()

    print("test_spec")
    test_spec()

    print("test_missing_lookup")
    test_missing_lookup()

    print("test_malformed_lookup")
    test_malformed_lookup()


    print("\n--- Tests complete ---\n")


if __name__ == "__main__":
    test_registration()
    




