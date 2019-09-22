import os
import numpy as np 
import os
import pandas as pd

from vitalutils import vitalfile
import biosppy
from keras.models import load_model

cur_path = os.path.dirname(os.path.abspath(__file__))

#########################################
# constant
_input_size = 256
_sampling_rate = 360 # MIT-BH db

PQ_DURATION = 0.20  # max PQ duration
QT_DURATION = 0.48  # max QT duration

_win1 = int(np.ceil(PQ_DURATION*_sampling_rate)) + 4
_win2 = int(np.ceil(QT_DURATION*_sampling_rate)) + 7

#########################################

class ArrhythmiaClassifier(object):

    def __init__(self):
        proj_dir = os.path.split(os.path.split(cur_path)[0])[0]
        model_path = os.path.join(proj_dir, 'models', 'arrhythmia_classifier.h5')
        self.model = load_model(model_path)
        self.classes = classes = np.array(['APC', 'LBB', 'NORMAL', 'PAB', 'PVC', 'RBB'])
        self.labeltoidx = {x: i-0 for i, x in enumerate(self.classes)}
        self.interval = (_win1 + _win2) / _sampling_rate


    def detect_durations(self, raw_signals, sampling_rate=500):        

        ret = np.zeros(len(self.classes))

        # exclude outlier signals at the begining and ending periods
        nskip = int(len(raw_signals)*0.1)
        raw_signals = raw_signals[nskip:-nskip]

        start_time = 0
        end_time = len(raw_signals)/sampling_rate
        x = np.linspace(start_time*1e3, end_time*1e3, int(end_time*_sampling_rate))
        x0 = np.linspace(start_time*1e3, end_time*1e3, int(end_time*sampling_rate)+1)
        x0 = x0[:len(raw_signals)]

        y = np.interp(x, x0, raw_signals)
        signals, params = biosppy.signals.tools.smoother(signal=y, kernel='boxzen', size=8)
        
        peaks = biosppy.signals.ecg.engzee_segmenter(signals, sampling_rate=500)[0]
        #peaks = biosppy.signals.ecg.christov_segmenter(signals, sampling_rate=500)[0]

        if len(peaks) < 10:
            return ret

        qrs_data = []
        features = []
        rep_rr_dist = np.median(peaks[1:] - peaks[:-1])
        prev_index = peaks[0]
        for i in range(1, len(peaks)-1):
            index = peaks[i]
            st = index - _win1
            ed = index + _win2
            if st >= 0 and ed < len(signals):
                y = signals[st:ed]
                rep_height = np.median(y)
                rr_dist1 = (index - prev_index)/rep_rr_dist
                rr_dist2 = (peaks[i+1] - index)/rep_rr_dist            
                
                denom = np.abs(np.max(y) - rep_height)
                if denom > 0:
                    peak_ratio = np.abs(rep_height - np.min(y)) / denom
                else:
                    peak_ratio = 0

                qrs_data.append(y)
                features.append([rr_dist1, rr_dist2, peak_ratio])
            prev_index = index


        # tmp_df = pd.DataFrame(features, columns=["prev_rr", "next_rr", "peak_ratio"])
        # qrs_df = pd.DataFrame(qrs_data)
        # qrs_df = pd.concat((qrs_df, tmp_df), axis=1)    

        #X_test = qrs_df.iloc[:, :input_size].values.reshape([-1, input_size, 1])
        #X_test2 = qrs_df.iloc[:, -3:].values

        X_qrs = np.stack(qrs_data, axis=0).reshape([-1, _input_size, 1])
        X_feat = np.stack(features, axis=0)

        y_prob = self.model.predict([X_qrs, X_feat])
        y_pred = np.argmax(y_prob, axis=1)

        labels, cnts = np.unique(y_pred, return_counts=True)
        #ret = {self.classes[label]:cnt*self.interval for label, cnt in zip(labels, cnts)}        

        
        for label, cnt in zip(labels, cnts):
            ret[label] = cnt*self.interval

        return ret












