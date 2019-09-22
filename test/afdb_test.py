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
'(AB':	'Atrial bigeminy',
'(AFIB':	'Atrial fibrillation',
'(AFL':	'Atrial flutter',
'(B':	'Ventricular bigeminy',
'(BII':	'2° heart block',
'(IVR':	'Idioventricular rhythm',
'(N':	'Normal sinus rhythm',
'(NOD':	'Nodal (A-V junctional) rhythm',
'(P':	'Paced rhythm',
'(PREX':	'Pre-excitation (WPW)',
'(SBR':	'Sinus bradycardia',
'(SVTA':	'Supraventricular tachyarrhythmia',
'(T':	'Ventricular trigeminy',
'(VFL':	'Ventricular flutter',
'(VT':	'Ventricular tachycardia'}

stats = {'(AFIB': 0, '(AFL': 0, '(J': 0, '(N': 0}

for fid, record_name in enumerate(paths):
    rid = int(record_name[-3:])
    raw_signals, fields = wfdb.rdsamp(record_name, channels=[0]) 
    ann = wfdb.rdann(record_name, 'atr')
    #labels, counts = np.unique(ann.aux_note, return_counts=True)
    for i in range(1, len(ann.sample)):
        label = ann.aux_note[i]
        if i == len(ann.sample)-1:
            duration = len(raw_signals) - ann.sample[i]
        else:
            duration = ann.sample[i+1] - ann.sample[i]
        stats[label] += duration


    #for label, index in zip(ann.aux_note, ann.sample):
for k, v in stats.items():
    print("{} : {}".format(k, v/(250*60)))

        
