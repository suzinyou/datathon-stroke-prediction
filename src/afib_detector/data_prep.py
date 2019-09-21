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
    
    for i in range(1, len(ann.sample)):
        label = ann.aux_note[i]
        st_index = ann.sample[i]
        ed_index = ann.sample[i+1] if i != len(ann.sample)-1 else len(signals)

        if label not in label2class:
            continue

        for index in range(st_index, ed_index, interval):
            y = signals[index:index+interval]
            if len(y) == interval:
                data.append(y)
                labels.append(label2class[label])
                rids.append(fid)


df = pd.DataFrame(data)
df['target'] = labels
df['record_id'] = rids
df.to_pickle('/home/jungkap/Documents/datathon/afib_data.pkl')



