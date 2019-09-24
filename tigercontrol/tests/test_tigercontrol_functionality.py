"""
Run all tests for the TigerControl framework
"""

import tigercontrol

# test all tigercontrol.* methods
def test_tigercontrol_functionality(show_results=False):
    print("\nrunning all tigercontrol functionality tests...\n")
    test_help()
    test_error()
    print("\nall tigercontrol functionality tests passed\n")


# test tigercontrol.help() method
def test_help():
    tigercontrol.help()


def test_error():
    try:
        from tigercontrol.error import Error
        raise Error()
    except Error:
        pass

if __name__ == "__main__":
    test_tigercontrol_functionality(show_results=False)