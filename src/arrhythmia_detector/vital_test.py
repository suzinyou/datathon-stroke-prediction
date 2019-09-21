import numpy as np 
import os
import pandas as pd

from vitalutils import vitalfile
from matplotlib import pylab as plt
import biosppy
from keras.models import load_model

model = load_model('/home/jungkap/Documents/datathon/model.h5')
classes = np.array(['APC', 'LBB', 'NORMAL', 'PAB', 'PVC', 'RBB'])
labeltoidx = {x: i-0 for i, x in enumerate(classes)}


#########################################
# constant
input_size = 256
PQ_DURATION = 0.20  # max PQ duration
QT_DURATION = 0.48  # max QT duration
sampling_rate = 360 # MIT-BH db
d1 = int(np.ceil(PQ_DURATION*sampling_rate)) + 4
d2 = int(np.ceil(QT_DURATION*sampling_rate)) + 7
#########################################

vital_sampling_rate = 500
vitaldb_path = '/home/jungkap/Documents/datathon/vitaldb'

result = dict()

for record_id in [814, 905, 2040, 4013, 5323, 5409, 3188, 2764, 264, 265]:
    if record_id in result:
        continue

    # fname = "{id:05d}.pkl".format(id=record_id)
    # df = pd.read_pickle(os.path.join(vitaldb_path, fname))

    ipath = os.path.join(vitaldb_path, '{id:05d}.vital'.format(id=record_id))
    vit = vitalfile.VitalFile(ipath, ['ECG_II'])
    raw_signals = vit.get_samples('ECG_II')

    start_time = 0
    end_time = len(raw_signals)/vital_sampling_rate
    x = np.linspace(start_time*1e3, end_time*1e3, int(end_time*sampling_rate))
    x0 = np.linspace(start_time*1e3, end_time*1e3, int(end_time*vital_sampling_rate)+1)
    x0 = x0[:len(raw_signals)]

    y = np.interp(x, x0, raw_signals)



    signals, params = biosppy.signals.tools.smoother(signal=y, kernel='boxzen', size=8)
    peaks = biosppy.signals.ecg.engzee_segmenter(signals, sampling_rate=500)[0]

    peaks = biosppy.signals.ecg.christov_segmenter(signals, sampling_rate=500)[0]


    qrs_data = []
    features = []
    rep_rr_dist = np.median(peaks[1:] - peaks[:-1])
    prev_index = peaks[0]
    for i in range(1, len(peaks)-1):
        index = peaks[i]
        st = index - d1
        ed = index + d2
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


    tmp_df = pd.DataFrame(features, columns=["prev_rr", "next_rr", "peak_ratio"])
    qrs_df = pd.DataFrame(qrs_data)
    qrs_df = pd.concat((qrs_df, tmp_df), axis=1)    



    X_test = qrs_df.iloc[:, :input_size].values.reshape([-1, input_size, 1])
    X_test2 = qrs_df.iloc[:, -3:].values


    y_prob = model.predict([X_test, X_test2])
    y_pred = np.argmax(y_prob, axis=1)


    predictions = classes[y_pred]
    labels, counts = np.unique(predictions, return_counts=True)

    stat = np.zeros(len(classes))
    for label, cnt in zip(labels, counts):
        stat[labeltoidx[label]] = cnt

    result[record_id] = stat








