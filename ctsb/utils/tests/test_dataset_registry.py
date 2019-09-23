import tigercontrol
import os
from tigercontrol.utils.dataset_registry import sp500, uci_indoor
from tigercontrol.utils.download_tools import get_tigercontrol_dir

# add all unit tests in datset_registry
def test_dataset_registry(show=False):
    # test_sp500_download()
    csv_paths = ['data/sp500.csv', 'data/uci_indoor_cleaned.csv']
    download_fns = [sp500, uci_indoor]
    for i in range (0, len(csv_paths)):
        test_dataset_download(csv_paths[i], download_fns[i], verbose=show)
    print("test_dataset_registry passed")

# Unit test for dataset_registry.download_fn. Tests if file is downloaded and processed correctly.
def test_dataset_download(path_to_csv, download_fn, verbose=False):
    download_fn(verbose)
    tigercontrol_dir = get_tigercontrol_dir()
    total_path = os.path.join(tigercontrol_dir, path_to_csv)
    assert(os.path.isfile(total_path))
    # os.remove(total_path)
    if verbose:
        name = path_to_csv.split('.')[0].split('/')[1]
        print(name + " download passed")

if __name__ == '__main__':
    test_dataset_registry(show=True)