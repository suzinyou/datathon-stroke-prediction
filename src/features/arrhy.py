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
import sys

if __name__ == "__main__":
    pid = int(sys.argv[1])

    flist = np.loadtxt('/home/jungkap/Documents/datathon/vitaldb/file.list', dtype=int)
    vitaldb_path = '/home/jungkap/Documents/datathon/vitaldb'
    #id_list = np.loadtxt('/home/jungkap/Documents/datathon/vitaldb/file.list', dtype=int)

    vital_files = glob('{}/*.vital'.format(vitaldb_path))

    arr_model = ArrhythmiaClassifier()
    afib_model = AFIBClassifier()

    filepath = '/home/jungkap/Documents/datathon/arrhythmia_feature.pkl'
    #features = dict()
    features = joblib.load(filepath)

    vital_files2 = []
    for ipath in vital_files:
        record_id = int(ipath[-11:-6])
        if record_id not in features and record_id in flist:
            vital_files2.append(ipath)
   
    vital_files = vital_files2
    print(len(vital_files2))
    np.random.seed(42)
    np.random.shuffle(vital_files)    

    n = int(len(vital_files)*0.2)
    sti = (pid - 1)*n
    edi = sti + n if pid < 5 else len(vital_files)

    filepath = '/home/jungkap/Documents/datathon/arrhythmia_feature{}.pkl'.format(pid)
    features = {}

    for ipath in vital_files[sti:edi]:
        record_id = int(ipath[-11:-6])
        ipath = os.path.join(vitaldb_path, '{id:05d}.vital'.format(id=record_id))

        if record_id in features or not os.path.exists(ipath):
            print('skip {}'.format(record_id))
            continue
        try:
            vit = vitalfile.VitalFile(ipath, ['ECG_II'])
            raw_signals = vit.get_samples2('ECG_II')
        
            ret1 = afib_model.detect_durations(raw_signals)
            ret2 = arr_model.detect_durations(raw_signals)
            features[record_id] = np.concatenate((ret1, ret2))
        except:
            print('Error in {}'.format(record_id))
            
        print(len(features))
        joblib.dump(features, filepath)

joblib.dump(features, filepath)


