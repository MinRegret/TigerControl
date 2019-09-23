from __future__ import division
from __future__ import print_function

import os
import sys
import zipfile
import tigercontrol
from urllib.error import URLError
from urllib.request import urlretrieve

def get_tigercontrol_dir():
    """
    Description:
        Gets absolute path of package directory.
    Returns:
        Absolute path of package directory
    """
    init_dir = os.path.abspath(tigercontrol.__file__)
    tigercontrol_dir = init_dir.rsplit('/', 1)[0]
    return tigercontrol_dir

def report_download_progress(chunk_number, chunk_size, file_size):
    """
    Description:
        Prints out the download progress bar.
    Args:
        chunk_number(int): chunk number
        chunk_size(int): maximize size of a chunk
        file_size(int): total size of download
    """
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = '#' * int(64 * percent)
        sys.stdout.write('\r0% |{:<64}| {}%'.format(bar, int(percent * 100)))

def download(destination_path, url, verbose):
    """
    Description:
        Downloads the file at url to destination_path.
    Args:
        destination_path(string): the destination path of the download
        url(string): the url of the file to download
        verbose(boolean): If True, will report download progress and inform if download already exists
    """
    if os.path.exists(destination_path):
        if verbose:
            print('{} already exists, skipping ...'.format(destination_path))
    else:
        if verbose:
            print('Downloading {} ...'.format(url))
        try:
            hook = report_download_progress if verbose else None
            urlretrieve(url, destination_path, reporthook=hook)
        except URLError:
            raise RuntimeError('Error downloading resource!')
        finally:
            if verbose:
                # Just a newline.
                print()

def unzip(zipped_path, unzipped_path=None, verbose=False):
    """
    Description:
        Unzips data from 'zipped_path'.
    Args:
        zipped_path(string): the path of the zipped file
        verbose(boolean): If True, will report download progress and inform if download already exists
    Returns:
        None
    """
    if not unzipped_path:
        unzipped_path = os.path.splitext(zipped_path)[0]
    if os.path.exists(unzipped_path):
        if verbose:
            print('{} already exists, skipping ... '.format(unzipped_path))
        return
    zip_ref = zipfile.ZipFile(zipped_path, 'r')
    zip_ref.extractall(unzipped_path)
    zip_ref.close()


