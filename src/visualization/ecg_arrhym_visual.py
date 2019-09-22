import wfdb
import numpy as np
from glob import glob
import biosppy
import os 
import cv2
import pandas as pd 

from matplotlib import pyplot as plt

mitdb_path = '/home/jungkap/Documents/datathon/alpha.physionet.org/files/mitdb/1.0.0/'
paths = glob('{}*.atr'.format(mitdb_path))
paths = [path[:-4] for path in paths]


symbol_abbr = ['APC', 'NORMAL', 'LBB', 'PVC', 'PAB', 'RBB']
symbol = ['A', 'N', 'L', 'V', '/', 'R']
symbol_map = {k: v for k, v in zip(symbol, symbol_abbr)}


# refer to ecg_annotator
PQ_DURATION = 0.20  # max PQ duration
QT_DURATION = 0.48  # max QT duration

sampling_rate = 360 # MIT-BH db

d1 = int(np.ceil(PQ_DURATION*sampling_rate)) + 4
d2 = int(np.ceil(QT_DURATION*sampling_rate)) + 7

for fid, record_name in enumerate(paths):
    rid = int(record_name[-3:])
    
    raw_signals, fields = wfdb.rdsamp(record_name, channels=[0]) 
    
    signals, params = biosppy.signals.tools.smoother(
        signal=raw_signals.squeeze(), kernel='boxzen', size=8)

    ann = wfdb.rdann(record_name, 'atr')
    rep_rr_dist = np.median(ann.sample[1:] - ann.sample[:-1])
    mask = np.in1d(ann.symbol, symbol)

    if np.sum(np.array(ann.symbol) == 'V') > 0:
        break

    for i in range(566, 571):
        label = ann.symbol[i]
        st_idx = ann.sample[i]
        ed_idx = ann.sample[i+1]
        print("{}-{}: {}".format(st_idx, ed_idx, label))
        x = np.arange(st_idx, ed_idx)
        if label == 'V':
            plt.plot(x, signals[x], c='r')
        else:
            plt.plot(x, signals[x], c='b')



np.where(np.array(ann.symbol) == 'V')