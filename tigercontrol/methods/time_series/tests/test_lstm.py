# test the LSTM method class

import tigercontrol
import numpy as onp
import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
from tigercontrol.utils import generate_key
from tigercontrol.utils.download_tools import get_tigercontrol_dir
import os
import pandas as pd
import ast

# https://cnsviewer.corp.google.com/cns/jn-d/home/floods/hydro_method/datasets/processed/full/
def test_flood_FL(steps=61, show_plot=True):
    T = steps 
    n, m, l, d = 6, 1, 3, 10
    # problem = tigercontrol.problem("LDS-Control-v0")
    # y_true = problem.initialize(n, m, d)
    tigercontrol_dir = get_tigercontrol_dir()
    data_path = os.path.join(tigercontrol_dir, 'data/FL_train.csv')
    df = pd.read_csv(data_path).head()
    # print(df['static:drain_area_log2'].head())
    # arr_results = onp.load(data_path, encoding='bytes')
    method = tigercontrol.method("LSTM")
    method.initialize(n, m, l, d)
    loss = lambda pred, true: np.sum(((pred - true)/true)**2)
    # num_batches = len(arr_results)
    # for i in range(num_batches):
    #    print("num_measurements:" + str(len(arr_results[i][2])))
    static_features_to_use=[
        ('static:drain_area_log2', False),
        ('train:site:mean:USGS:discharge_mean', False),
        ('train:site:std:USGS:discharge_mean', False),
    ]

    sequence_features_to_use=[
        ('sequence:GSMAP_MERGED:hourlyPrecipRate', False),
        ('sequence:GLDAS21:Tair_f_inst', True),
        ('sequence:AQUA_VI:NDVI', True)
    ]

    label_feature_name = 'label:USGS:discharge_mean'
    past_sequence_label_feature_name = 'sequence:USGS:discharge_mean'
    
    # print("num_batches: " + str(num_batches))
    feature_list = []
    # sequence_length = df['sequence:USGS:discharge_mean_shape']
    # print("type(sequence_length):" + str(type(sequence_length)))
    # print("sequence length: " + sequence_length)
    sequence_length = 61
    label_list = []
    for i in range(5):
        for j in range(sequence_length):
            feature = []
            for (seq_feat, b) in sequence_features_to_use:
                # print("seq_feat: " + seq_feat)
                # print("df[seq_feat].shape: " + str(df[seq_feat].shape))
                # print(df[seq_feat])
                # print(type(df[seq_feat].iloc[i]))
                # print(ast.literal_eval(df[seq_feat].iloc[i]))
                feature.append(ast.literal_eval(df[seq_feat].iloc[i])[j])
            for (stat_feat, c) in static_features_to_use:
                feature.append(ast.literal_eval(df[stat_feat].iloc[0])[0])
            feature_list.append(feature)
            label_list.append(ast.literal_eval(df[label_feature_name].iloc[0])[0])
    print(feature_list[0])
    print(len(feature_list[0]))
    print(len(feature_list))
    print(len(label_list))
    print(label_list[:10])
    results = []
    for i in range(len(feature_list)):
        u = feature_list[i]
        y_pred = method.predict(u)
        y_true = label_list[i]
        print("y_pred: " + str(y_pred))
        print("y_true: " + str(y_true))
        results.append(loss(y_true, y_pred))
        method.update(y_true)
    print(results[-10:])
    
    if show_plot:
        plt.plot(results[25:])
        plt.title("LSTM method on FL_train data")
        plt.show(block=True)
        plt.close()


    '''
    for i in range(1000):
        print(i)
        for j in range(61):
            y_pred = method.predict(arr_results[i][2][j])
            y_true = arr_results[i][3][j]
            results.append(loss(y_true, y_pred))
            method.update(y_true)'''
    '''
    if show_plot:
        plt.plot(results)
        plt.title("LSTM method on LDS problem")
        plt.show(block=True)
        plt.close()
    print("test_lstm passed")'''
    return

def test_lstm(steps=100, show_plot=True):
    T = steps 
    n, m, l, d = 4, 5, 10, 10
    problem = tigercontrol.problem("LDS-Control-v0")
    y_true = problem.initialize(n, m, d)
    method = tigercontrol.method("LSTM")
    method.initialize(n, m, l, d)
    loss = lambda pred, true: np.sum((pred - true)**2)
 
    results = []
    for i in range(T):
        u = random.normal(generate_key(), (n,))
        y_pred = method.predict(u)
        y_true = problem.step(u)
        results.append(loss(y_true, y_pred))
        method.update(y_true)

    if show_plot:
        plt.plot(results)
        plt.title("LSTM method on LDS problem")
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    print("test_lstm passed")
    return

if __name__=="__main__":
    # test_lstm()
    test_flood_FL()