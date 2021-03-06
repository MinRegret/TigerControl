# test registration_tools

import os
import re

import tigercontrol
from tigercontrol import error, environments
from tigercontrol.utils.registration_tools import *

# run all registry tests
def test_registration_tools(show=False):
    test_registry()
    test_tigercontrol_environment()
    test_missing_lookup()
    print("test_registration_tools passed")


# add all unit tests in datset_registry
def test_registry():
    regexp = re.compile(r'^([\w:.-]+)$') # regular expression accepts "string"-v#
    test_registry = Registry(regexp)

    test_registry.register(id='GoodID', entry_point='tigercontrol.environments:LDS')
    try:
        test_registry.register(id='GoodID', entry_point='tigercontrol.environments:LDS')
        raise Exception("Registry successfully registered bad ID")
    except error.Error:
        pass
    keys = test_registry.list_ids()
    vals = test_registry.all()
    obj_class = test_registry.get_class('GoodID')
    obj_instance = obj_class()
    return


def test_tigercontrol_environment():
    environment = tigercontrol.environment('LDS')
    return


def test_missing_lookup():
    environment_id_re = re.compile(r'^(?:[\w:-]+\/)?([\w:.-]+)$')
    registry = Registry(environment_id_re)
    registry.register(id='Test', entry_point=None)
    registry.register(id='Test-v15', entry_point=None)
    registry.register(id='Test-v9', entry_point=None)
    registry.register(id='Other-v100', entry_point=None)

    try:
        registry.spec('Unknown-v1')
    except error.UnregisteredObject:
        pass
    else:
        assert False


if __name__ == '__main__':
    test_registration_tools()

