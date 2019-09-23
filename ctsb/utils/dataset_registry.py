from __future__ import division
from __future__ import print_function

import os
import shutil
import xlrd
import datetime
import csv
import pandas as pd
import numpy as np
from tigercontrol.utils.download_tools import *


def to_datetime(date, time):
    """ 
    Description: Takes a date and a time and converts it to a datetime object.
    Args:
        date (string): Date in DD/MM/YYYY format
        time (string): Time in hh:mm format
    Returns:
        Datetime object containing date and time information
    """
    day_month_year = [int(x) for x in date.split('/')]
    hour_min = [int(x) for x in time.split(':')]

    return datetime.datetime(day_month_year[2], 
                             day_month_year[1], 
                             day_month_year[0], 
                             hour_min[0], 
                             hour_min[1])

def datetime_to_daysElapsed(cur_datetime, base_datetime):
    """
    Description:
        Computes the number of days elapsed since 'base' date.

    Args:
        cur_datetime (datetime): Current date and time
        base_datetime (datetime): Base date and time
    Returns:
        Datetime object containing number of days elapsed
    """
    time_delta = cur_datetime - base_datetime
    time_to_days = (time_delta.seconds)/(24 * 60 * 60)
    return time_delta.days + time_to_days

# Dataset credits: https://fred.stlouisfed.org/series/UNRATE, 
# Federal Reserve Bank of St. Louis.

def unemployment(verbose=True):
    """ 
    Description: Checks if unemployment data exists, downloads if not.
        Dataset credits: https://fred.stlouisfed.org/series/UNRATE, 
        Federal Reserve Bank of St. Louis.

    Args:
        verbose (boolean): Specifies if download progress should be printed
    Returns:
        Dataframe containing Unemployment data
    """

    import datetime
    from urllib.request import urlretrieve
    tigercontrol_dir = get_tigercontrol_dir()
    path_csv = os.path.join(tigercontrol_dir, 'data/unemployment.csv')
    if os.path.exists(path_csv): # if file exists return csv
        return pd.read_csv(path_csv)


    day = datetime.date.today()
    for i in range(10): # try downloading 10 times
        day_string = day.isoformat()
        url_core = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=UNRATE&scale=left&cosd=1948-01-01&coed=2019-06-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2009-06-01&line_index=1&transformation=lin&vintage_date={}&revision_date={}&nd=1948-01-01"
        url_csv = url_core.format(day_string, day_string)

        try:
            urlretrieve(url_csv, path_csv)
            return pd.read_csv(path_csv)
        except Exception as e:
            if verbose:
                print("date {} download failed: {}".format(day, e))

        day = day - datetime.timedelta(days=1) # try with previous day

    # final attempt
    try:
        url_csv = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=UNRATE&scale=left&cosd=1948-01-01&coed=2019-06-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2009-06-01&line_index=1&transformation=lin&vintage_date=2019-07-19&revision_date=2019-07-19&nd=1948-01-01"

        urlretrieve(url_csv, path_csv)
        return pd.read_csv(path_csv)
    except Exception as e:
        raise RuntimeError("Error downloading URL: {}".format(e))


def uci_indoor(verbose=True):
    """
    Description: Checks if uci_indoor data exists, downloads if not. 
        Dataset credits: F. Zamora-Martínez, P. Romeu, P. Botella-Rocamora, J. Pardo, 
        On-line learning of indoor temperature forecasting models towards energy efficiency,
        Energy and Buildings, Volume 83, November 2014, Pages 162-172, ISSN 0378-7788

    Args:
        verbose (boolean): Specifies if download progress should be printed
    Returns:
        Dataframe containing uci_indoor data
    """

    tigercontrol_dir = get_tigercontrol_dir()
    url_uci_indoor = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00274/NEW-DATA.zip'
    path_uci_indoor_zip = os.path.join(tigercontrol_dir, 'data/uci_indoor.zip')
    path_uci_indoor_unzip = os.path.join(tigercontrol_dir, 'data/uci_indoor')
    path_uci_indoor_txt1 = os.path.join(tigercontrol_dir, 'data/uci_indoor/NEW-DATA-1.T15.txt')
    path_uci_indoor_csv = os.path.join(tigercontrol_dir, 'data/uci_indoor.csv')
    path_uci_indoor_cleaned = os.path.join(tigercontrol_dir, 'data/uci_indoor_cleaned.csv')

    # check if files have been downloaded before, else download
    if not os.path.exists(path_uci_indoor_cleaned):
        download(path_uci_indoor_zip, url_uci_indoor, verbose) # get files from online URL
        unzip(path_uci_indoor_zip)
        os.remove(path_uci_indoor_zip) # clean up - remove unnecessary files
        
        with open(path_uci_indoor_txt1, 'r') as txt_file:
            list_of_vecs = [line.split() for line in txt_file] # clean downloaded data
            list_of_vecs[0] = list_of_vecs[0][1:]
        with open(path_uci_indoor_csv, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(list_of_vecs)
        shutil.rmtree(path_uci_indoor_unzip) # clean up - remove unnecessary files

        df = pd.read_csv(path_uci_indoor_csv)
        base_datetime = to_datetime(df['1:Date'].iloc[0], df['2:Time'].iloc[0])

        def uci_datetime_converter(row):
            return datetime_to_daysElapsed(to_datetime(row['1:Date'],row['2:Time']), base_datetime)
        
        df['24:Day_Of_Week'] = df.apply(uci_datetime_converter, axis=1)
        with open(path_uci_indoor_csv,'r') as csvinput:
            with open(path_uci_indoor_cleaned, 'w') as csvoutput:
                writer = csv.writer(csvoutput, lineterminator='\n')
                reader = csv.reader(csvinput)
                r = 0
                appended_csv = [next(reader) + ['25:Days_Elapsed']]
                for row in reader:
                    row.append(df['24:Day_Of_Week'].iloc[r])
                    appended_csv.append(row)
                    r += 1
                writer.writerows(appended_csv)
        os.remove(path_uci_indoor_csv)
        
    df = pd.read_csv(path_uci_indoor_cleaned) # clean up - remove unnecessary files
    return df


def sp500(verbose=True):
    """
    Description: Checks if S&P500 data exists, downloads if not.

    Args:
        verbose (boolean): Specifies if download progress should be printed
    Returns:
        Dataframe containing S&P500 data
    """

    tigercontrol_dir = get_tigercontrol_dir()
    url_sp500_xls = 'http://www.cboe.com/micro/buywrite/dailypricehistory.xls'
    path_sp500_xls = os.path.join(tigercontrol_dir, 'data/sp500_xls.xls')
    path_sp500_txt = os.path.join(tigercontrol_dir, 'data/sp500_col.txt')
    path_sp500_csv = os.path.join(tigercontrol_dir, 'data/sp500.csv')

    # check if files have been downloaded before, else download
    if not os.path.exists(path_sp500_csv):
        download(path_sp500_xls, url_sp500_xls, verbose) # get files from online URL
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


def crypto():
    """
    Description: Checks if cryptocurrency data exists, downloads if not.

    Args:
        None
    Returns:
        Dataframe containing cryptocurrency data
    """
    tigercontrol_dir = get_tigercontrol_dir()
    path_crypto_csv = os.path.join(tigercontrol_dir, 'data/crypto.csv')
    
    if not os.path.exists(path_crypto_csv):
        df = pd.read_csv('https://query.data.world/s/43quzwdjeh2zmghpdcgvgkppo6bvg7')
        dict_of_currency_dfs = {k: v for k, v in df.groupby('Currency')}
        bdf = dict_of_currency_dfs['bitcoin']
        bdf.to_csv(path_crypto_csv)
    return pd.read_csv(path_crypto_csv)


def enso(input_signals, include_month, output_signals, history, timeline):
    """
    Description: Transforms the ctrl_indices dataset into a format suitable for online learning.

    Args:
        input_signals (list of strings): signals used for prediction
        include_month (boolean): True if the month should be used as a feature,
                                 False otherwise
        output_signals (list of strings): signals we are trying to predict
        history (int): number of past observations used for prediction
        timeline (int/list of ints): the forecasting timeline(s)
    Returns:
        X (numpy.ndarray): Input Observations
        y (numpy.ndarray): Labels
    """

    ############################## GET DATA ###################################

    tigercontrol_dir = get_tigercontrol_dir()
    datapath = os.path.join(tigercontrol_dir, 'data/enso.csv')
    signals_pd = pd.read_csv(datapath)

    signal_length = signals_pd['nino34'].values.shape[0]

    # make timeline into np.array
    if(type(timeline) is int):
        timeline = [timeline]

    timeline = np.array(timeline)

    ################################ GET ONI ###################################

    ''' 0. Get nino34 signal for ONI'''
    nino34 = signals_pd['nino34'].values

    ''' 1. Get climatology '''
    clm = np.zeros(12)
    for month in range(12):
        section = [12 * i + month for i in range(signal_length // 12)]
        clm[month] = np.mean(nino34[section])


    ''' 2. Compute anomaly '''
    anm = np.array(nino34)
    for i in range(signal_length):
        anm[i] = nino34[i] - clm[i % 12]

    ''' 3. Get ONI '''
    oni = np.array(anm)
    m = 3
    for i in range(signal_length):
        oni[i] = np.mean(anm[max(0, (i - m + 1)) : min((i + 1), signal_length)])

    ########################## PREPARE INPUT/ OUTPUT SIGNALS ############################

    # input signals
    X_signals = np.zeros((signal_length, 0))

    for signal in input_signals:
        if(signal == 'oni'):
            new_signal = oni.reshape((signal_length, 1))
        else:
            new_signal = (signals_pd[signal].values).reshape((signal_length, 1))
        X_signals = np.append(X_signals, new_signal, axis = 1)

    if(include_month):
        new_signal = (np.arange(signal_length) % 12).reshape((signal_length, 1))
        X_signals = np.append(X_signals, new_signal, axis = 1)

    # output signals
    y_signals = np.zeros((signal_length, 0))

    for signal in output_signals:
        if(signal == 'oni'):
            new_signal = oni.reshape((signal_length, 1))
        else:
            new_signal = (signals_pd[signal].values).reshape((signal_length, 1))
        y_signals = np.append(y_signals, new_signal, axis = 1)

    ########################## CONVERT SIGNALS TO (X, y) #############################

    effective_length = signal_length - history - np.max(timeline)

    # Observations
    X = np.ndarray((effective_length, history, np.array(input_signals).shape[0] + include_month))

    for i in range(effective_length):
        X[i, 0:history, :] = X_signals[i:(i + history)]

    # if we use only one observation
    if(history == 1):
        X = X.reshape((effective_length, np.array(input_signals).shape[0] + include_month))

    # Labels
    y = np.ndarray((effective_length, timeline.shape[0], np.array(output_signals).shape[0]))

    n_t = 0
    for t in timeline:
        for i in range(effective_length):
            y[i, n_t, :] = y_signals[i + history + t - 1]
        n_t += 1

    # if we predict only one timeline
    if(timeline.shape[0] == 1):
        y = y.reshape((effective_length, np.array(output_signals).shape[0]))

    return (X, y)
    