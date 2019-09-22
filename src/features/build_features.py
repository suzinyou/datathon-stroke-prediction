import numpy as np 
import os
import pandas as pd

from vitalutils import vitalfile
from matplotlib import pylab as plt
import biosppy
from keras.models import load_model
from arrhythmia_detector.predictor import ArrhythmiaClassifier
from afib_detector.predictor import AFIBClassifier
import joblib
from glob import glob


vitaldb_path = '/home/jungkap/Documents/datathon/vitaldb'
#id_list = np.loadtxt('/home/jungkap/Documents/datathon/vitaldb/file.list', dtype=int)

vital_files = glob('{}/*.vital'.format(vitaldb_path))

arr_model = ArrhythmiaClassifier()
afib_model = AFIBClassifier()

filepath = '/home/jungkap/Documents/datathon/arrhythmia_feature.pkl'
#features = dict()
features = joblib.load(filepath)

for i in range(1, 5):
    filepath = '/home/jungkap/Documents/datathon/arrhythmia_feature{}.pkl'.format(i)
    tmp = joblib.load(filepath)
    for k, v in tmp.items():
        features[k] = v




for ipath in vital_files:
    record_id = int(ipath[-11:-6])
    ipath = os.path.join(vitaldb_path, '{id:05d}.vital'.format(id=record_id))
    print(record_id)

    if record_id in features or not os.path.exists(ipath):
        print('skip {}'.format(record_id))
        continue

    vit = vitalfile.VitalFile(ipath, ['ECG_II'])
    raw_signals = vit.get_samples2('ECG_II')
    
    ret1 = afib_model.detect_durations(raw_signals)
    ret2 = arr_model.detect_durations(raw_signals)

    features[record_id] = np.concatenate((ret1, ret2))
    print(len(features))
    joblib.dump(features, filepath)


joblib.dump(features, filepath)


