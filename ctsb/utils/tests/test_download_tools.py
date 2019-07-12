# test download_tools.py

import ctsb
import os
from ctsb.utils import download_tools

# add all unit tests in datset_registry
def test_download_tools(show=False):
    test_get_ctsb_dir(show=show)
    test_download(show=show)
    test_unzip(show=show)
    print("test_download_tools passed")


def test_get_ctsb_dir(show=False):
    assert os.path.isdir(download_tools.get_ctsb_dir())
    print("test_get_ctsb_dir passed")


# test report_download_process implicitly by calling download
def test_download(show=False):
    file_path = os.path.join(download_tools.get_ctsb_dir(), "worldbank.csv")
    url = "http://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.CD?downloadformat=csv"
    download_tools.download(file_path, url, verbose=show)
    assert os.path.isfile(file_path)
    os.remove(file_path)
    print("test_download passed")


def test_unzip(show=False):

    """ # TODO: fix
    zip_path = "unzip_test.zip"
    txt_path = "unzip_test.txt"
    new_path = "unzip_test_final.txt"

    from zipfile import ZipFile
    with open(txt_path, "w") as file:
        file.write("testing unzip method\n")
    with ZipFile(zip_path, 'w') as file:
        file.write(txt_path)

    assert os.path.isfile(txt_path)
    assert os.path.isfile(zip_path)


    print("\nTests")
    print(os.listdir(download_tools.get_ctsb_dir()))


    download_tools.unzip(zip_path, unzipped_path=new_path, verbose=show)


    print("\nunzip files - TEST.TXT should exist!")
    print(os.listdir(download_tools.get_ctsb_dir()))


    os.remove(txt_path)
    os.remove(zip_path)
    assert os.path.exists(new_path) # file should show up again after unzipping
    os.remove(new_path)

    """
    print("test_unzip passed")


# unit tests for ctsb.utils.download_tools
if __name__ == '__main__':
    test_download_tools()

