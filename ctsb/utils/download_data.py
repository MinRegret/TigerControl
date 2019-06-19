from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
from urllib import request
from urllib.error import URLError
from urllib.request import urlretrieve
import xlrd
import datetime
import csv

def report_download_progress(chunk_number, chunk_size, file_size):
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = '#' * int(64 * percent)
        sys.stdout.write('\r0% |{:<64}| {}%'.format(bar, int(percent * 100)))


def download(destination_path, url, quiet):
    if os.path.exists(destination_path):
        if not quiet:
            print('{} already exists, skipping ...'.format(destination_path))
    else:
        print('Downloading {} ...'.format(url))
        try:
            hook = None if quiet else report_download_progress
            urlretrieve(url, destination_path, reporthook=hook)
        except URLError:
            raise RuntimeError('Error downloading resource!')
        finally:
            if not quiet:
                # Just a newline.
                print()

def download_csv(csv_url, dest):
	csv = request.urlopen(csv_url)
	csv_data = csv.read()
	csv_str = str(csv_data)
	file = csv_str.split('\\n')
	dest_url = dest
	wr = open(dest_url, 'w')
	for data in file:
	    wr.write(data + '\n')
	wr.close()


def txt_to_csv(path_txt, path_csv):
	with open(path_txt) as f:
		with open(path_csv,'w') as out:
			csv_out=csv.writer(out)
			csv_out.writerow(['date','value'])
			for x in f.readlines():
				date_val_list = x.strip().split(',')
				date_val_list[0] = (date_val_list[0].split(' '))[0]
				csv_out.writerow(date_val_list)
	
def main():
	url_sp500_xls = 'http://www.cboe.com/micro/buywrite/dailypricehistory.xls'
	path_sp500_xls = '../data/sp500_xls.xls'
	path_sp500_txt = '../data/sp500_col.txt'
	path_sp500_csv = '../data/sp500.csv'
	if not os.path.exists(path_sp500_xls):
		download(path_sp500_xls, url_sp500_xls, False)
		book = xlrd.open_workbook(path_sp500_xls)
		sh = book.sheet_by_index(0)
		sp500_col = open(path_sp500_txt, 'w')
		for r in range(5, 8197):
			date = datetime.datetime(*xlrd.xldate_as_tuple(sh.cell_value(r,0), book.datemode))
			sp500_col.write(str(date) + "," + str(sh.cell(r, 3).value)+"\n")
		sp500_col.close()
	txt_to_csv(path_sp500_txt, path_sp500_csv)

if __name__ == '__main__':
    main()