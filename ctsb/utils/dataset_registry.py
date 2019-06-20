from __future__ import division
from __future__ import print_function

import os
import xlrd
import datetime
import csv
import pandas as pd
from ctsb.utils.download_tools import *

# check if S&P500 data exists, downloads if not. Returns dataframe containing data
def sp500():
	ctsb_dir = get_ctsb_dir()
	url_sp500_xls = 'http://www.cboe.com/micro/buywrite/dailypricehistory.xls'
	path_sp500_xls = os.path.join(ctsb_dir, 'data/sp500_xls.xls')
	path_sp500_txt = os.path.join(ctsb_dir, 'data/sp500_col.txt')
	path_sp500_csv = os.path.join(ctsb_dir, 'data/sp500.csv')

	# check if files have been downloaded before, else download
	if not os.path.exists(path_sp500_csv):
		download(path_sp500_xls, url_sp500_xls, False) # get files from online URL
		book = xlrd.open_workbook(path_sp500_xls)
		sh = book.sheet_by_index(0)
		sp500_col = open(path_sp500_txt, 'w')
		for r in range(5, 8197):
			date = datetime.datetime(*xlrd.xldate_as_tuple(sh.cell_value(r,0), book.datemode))
			sp500_col.write(str(date) + "," + str(sh.cell(r, 3).value)+"\n")
		sp500_col.close()
		with open(path_sp500_txt) as f: # clean downloaded data
			with open(path_sp500_csv,'w') as out:
				csv_out=csv.writer(out)
				csv_out.writerow(['date','value'])
				for x in f.readlines():
					date_val_list = x.strip().split(',')
					date_val_list[0] = (date_val_list[0].split(' '))[0]
					csv_out.writerow(date_val_list)
		os.remove(path_sp500_xls) # clean up - remove unnecessary files
		os.remove(path_sp500_txt)
	return pd.read_csv(path_sp500_csv)
