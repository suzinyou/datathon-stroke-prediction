import numpy as np 
import os
import pandas as pd

from vitalutils import vitalfile
from afib_detector.predictor import AFIBClassifier



record_id = 814
vitaldb_path = '/home/jungkap/Documents/datathon/vitaldb'

ipath = os.path.join(vitaldb_path, '{id:05d}.vital'.format(id=record_id))
vit = vitalfile.VitalFile(ipath, ['ECG_II'])
raw_signals = vit.get_samples('ECG_II')

model = AFIBClassifier()
ret = model.detect_durations(raw_signals)




# # downsampling
# raw_signals = raw_signals[range(0, len(raw_signals), 2)]
# signals, params = biosppy.signals.tools.smoother(signal=raw_signals, kernel='boxzen', size=8)
# data = [signals[index:index+interval] for index in range(0, len(signals)-interval, interval)]
# in_data = np.stack(data, axis=0).reshape(data.shape[0], interval, 1)

# y_prob = model.predict(in_data)
# y_pred = np.argmax(y_prob, axis=1)





