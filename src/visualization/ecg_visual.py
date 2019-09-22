import wfdb
import numpy as np
from glob import glob
import biosppy
import os 
import cv2
import pandas as pd 

wfdb.io.show_ann_labels()

from matplotlib import pyplot as plt

mitdb_path = '/home/jungkap/Documents/datathon/alpha.physionet.org/files/afdb/1.0.0/'
paths = glob('{}*.atr'.format(mitdb_path))
paths = [path[:-4] for path in paths]

class_map = {
    '(AFIB':	'Atrial fibrillation',
    '(AFL':	'Atrial flutter',
    '(N':	'Normal sinus rhythm'
}
label2class = {'(N': 'NORMAL', '(AFIB': 'AFIB', '(AFL': 'AF'}

stats = {k: 0 for k in class_map.keys()}
duration = 5 # seconds
sample_rate = 250 # hz
interval = int(duration * sample_rate)

data = []
labels = []
rids = []
for fid, record_name in enumerate(paths):
    rid = int(record_name[-3:])
    print(rid)
    raw_signals, fields = wfdb.rdsamp(record_name, channels=[0]) 
    ann = wfdb.rdann(record_name, 'atr')

    signals, params = biosppy.signals.tools.smoother(
        signal=raw_signals.squeeze(), kernel='boxzen', size=8)    

    
from matplotlib import pyplot as plt

for i in range(2, 5):
    label = ann.aux_note[i]
    st_idx = ann.sample[i]
    ed_idx = ann.sample[i+1]
    print("{}-{}".format(st_idx, ed_idx))
    x = np.arange(st_idx, ed_idx)
    if label == '(AFIB':
        plt.plot(x, signals[x], c='r')
    else:
        plt.plot(x, signals[x], c='b')
    
    