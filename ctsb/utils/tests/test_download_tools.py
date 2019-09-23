# test download_tools.py

import tigercontrol
import os
from tigercontrol.utils import download_tools

# add all unit tests in datset_registry
def test_download_tools(show=False):
    test_get_tigercontrol_dir(show=show)
    test_download(show=show)
    print("test_download_tools passed")


def test_get_tigercontrol_dir(show=False):
    assert os.path.isdir(download_tools.get_tigercontrol_dir())
    print("test_get_tigercontrol_dir passed")


# test report_download_process implicitly by calling download
def test_download(show=False):
    file_path = os.path.join(download_tools.get_tigercontrol_dir(), "worldbank.csv")
    url = "http://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.CD?downloadformat=csv"
    download_tools.download(file_path, url, verbose=show)
    assert os.path.isfile(file_path)
    os.remove(file_path)
    print("test_download passed")


# unit tests for tigercontrol.utils.download_tools
if __name__ == '__main__':
    test_download_tools()

