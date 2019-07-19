"""
Run all tests for the CTSB framework
"""

import ctsb

# test all ctsb.* methods
def test_ctsb_functionality(show_results=False):
    print("\nrunning all ctsb functionality tests...\n")
    test_help()
    test_error()
    print("\nall ctsb functionality tests passed\n")


# test ctsb.help() method
def test_help():
	ctsb.help()


def test_error():
	try:
    	from ctsb.error import Unregistered
    	raise Unregistered()
    except Unregistered:
    	pass

if __name__ == "__main__":
    test_ctsb_functionality(show_results=False)