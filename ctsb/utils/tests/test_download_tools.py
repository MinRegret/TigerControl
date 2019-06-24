# test download_tools.py

import ctsb
import os
from ctsb.utils import download_tools

# add all unit tests in datset_registry
def test_download_tools():
    test_ctsb_dir()
    print("test_download_tools passed")


# unit test for get_ctsb_dir
def test_ctsb_dir():
    assert os.path.isdir(download_tools.get_ctsb_dir())
    print("test_ctsb_dir passed")


# Unit test for ctsb.utils.download_tools.get_ctsb_dir
if __name__ == '__main__':
    test_download_tools()