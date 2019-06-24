import os
import re
import ctsb
from ctsb import error
from ctsb.utils.registration_tools import *

# add all unit tests in datset_registry
def test_registration_tools():
    regexp = re.compile(r'^([\w:.-]+)-v(\d+)$') # regular expression accepts "string"-v#
    test_registry = Registry(regexp)

    test_registry.register(id='GoodID-v0', entry_point='ctsb.problems.simulated:Random')
    try:
        test_registry.register(id='BadID', entry_point='ctsb.problems.simulated:Random')
        raise Exception("Registry successfully registered bad ID")
    except error.Error:
        pass
    keys = test_registry.keys()
    vals = test_registry.all()
    print("test_registration_tools passed")

if __name__ == '__main__':
    test_registration_tools()