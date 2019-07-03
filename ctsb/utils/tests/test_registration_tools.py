# test registration_tools

import os
import re

import ctsb
from ctsb import error, problems
from ctsb.utils.registration_tools import *
from ctsb.problems.registration import ProblemRegistry


def test_registration_tools():
    test_registry()
    test_ctsb_problem()
    test_missing_lookup()
    print("test_registration_tools passed")


# add all unit tests in datset_registry
def test_registry():
    regexp = re.compile(r'^([\w:.-]+)-v(\d+)$') # regular expression accepts "string"-v#
    test_registry = Registry(regexp)

    test_registry.register(id='GoodID-v0', entry_point='ctsb.problems.control:Random')
    try:
        test_registry.register(id='BadID', entry_point='ctsb.problems.control:Random')
        raise Exception("Registry successfully registered bad ID")
    except error.Error:
        pass
    keys = test_registry.list_ids()
    vals = test_registry.all()


def test_ctsb_problem():
    problem = ctsb.problem('Random-v0')
    assert problem.spec.id == 'Random-v0'
    return


def test_missing_lookup():
    problem_id_re = re.compile(r'^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$')
    registry = ProblemRegistry(problem_id_re)
    registry.register(id='Test-v0', entry_point=None)
    registry.register(id='Test-v15', entry_point=None)
    registry.register(id='Test-v9', entry_point=None)
    registry.register(id='Other-v100', entry_point=None)
    try:
        registry.spec('Test-v1')  # must match an problem name but not the version above
    except error.DeprecatedObject:
        pass
    else:
        assert False

    try:
        registry.spec('Unknown-v1')
    except error.UnregisteredObject:
        pass
    else:
        assert False


if __name__ == '__main__':
    test_registration_tools()

