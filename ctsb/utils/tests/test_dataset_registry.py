import ctsb
import os
from ctsb.utils.dataset_registry import sp500
from ctsb.utils.download_tools import get_ctsb_dir

# add all unit tests in datset_registry
def test_dataset_registry():
    test_sp500_download()
    print("test_dataset_registry passed")

# Unit test for dataset_registry.SP500. Tests if file is downloaded and processed correctly.
def test_sp500_download(quiet=True):
    sp500(quiet)
    ctsb_dir = get_ctsb_dir()
    path_sp500_csv = os.path.join(ctsb_dir, 'data/sp500.csv')
    assert(os.path.isfile(path_sp500_csv))
    os.remove(path_sp500_csv)
    print("test_sp500_download passed")


if __name__ == '__main__':
    test_dataset_registry()