import ctsb
import os
from ctsb.utils.dataset_registry import sp500
from ctsb.utils.download_tools import get_ctsb_dir


def test_sp500_download():
    sp500()
    ctsb_dir = get_ctsb_dir()
    path_sp500_xls = os.path.join(ctsb_dir, 'data/sp500_xls.xls')
    path_sp500_txt = os.path.join(ctsb_dir, 'data/sp500_col.txt')
    path_sp500_csv = os.path.join(ctsb_dir, 'data/sp500.csv')
    paths = [path_sp500_xls, path_sp500_txt, path_sp500_csv]
    for p in paths:
        assert(os.path.isfile(p))
        print("{} downloaded successfully".format(p))
        os.remove(p)


if __name__ == '__main__':
    test_sp500_download()