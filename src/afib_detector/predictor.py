import os
import numpy as np 
import os
import pandas as pd

from vitalutils import vitalfile
import biosppy
from keras.models import load_model

cur_path = os.path.dirname(os.path.abspath(__file__))
_duration = 5 # seconds
_sampling_rate = 250 # sampling rate given by mit-bh db

class AFIBClassifier(object):

    def __init__(self):
        proj_dir = os.path.split(os.path.split(cur_path)[0])[0]
        model_path = os.path.join(proj_dir, 'models', 'afib_classifier.h5')
        self.model = load_model(model_path)
        self.classes = np.array(['AF', 'AFIB', 'NORMAL'])
        self.labeltoidx = {x: i for i, x in enumerate(self.classes)}
        self.interval = _duration*_sampling_rate

    def detect_durations(self, raw_signals, sampling_rate=500):
        
        interval = self.interval
        ret = np.zeros(len(self.classes))

        if sampling_rate == 500:
            raw_signals = raw_signals[range(0, len(raw_signals), 2)]
        else:
            start_time = 0
            end_time = len(raw_signals)/sampling_rate
            x = np.linspace(start_time*1e3, end_time*1e3, int(end_time*_sampling_rate))
            x0 = np.linspace(start_time*1e3, end_time*1e3, int(end_time*sampling_rate)+1)
            x0 = x0[:len(raw_signals)]
            raw_signals = np.interp(x, x0, raw_signals)

        signals, params = biosppy.signals.tools.smoother(signal=raw_signals, kernel='boxzen', size=8)
        data = [
            signals[index:index+interval] \
                for index in range(0, len(signals)-interval, interval)
        ]

        n = len(data)
        in_data = np.stack(data, axis=0).reshape(n, interval, 1)

        y_prob = self.model.predict(in_data)
        y_pred = np.argmax(y_prob, axis=1)

        labels, cnts = np.unique(y_pred, return_counts=True)
        # ret = {self.classes[label]:cnt*interval for label, cnt in zip(labels, cnts)}
        
        for label, cnt in zip(labels, cnts):
            ret[label] = cnt*self.interval

        return ret







