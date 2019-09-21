import numpy as np 
import os
import pandas as pd

from vitalutils import vitalfile
from matplotlib import pylab as plt
import biosppy

sampling_rate = 500

vitaldb_path = '/home/jungkap/Documents/datathon/vitaldb'
record_id = 265

filepath = os.path.join(vitaldb_path, '{id:05d}.vital'.format(id=record_id))

vit = vitalfile.VitalFile(filepath, ['ECG_II'])
raw_signals = vit.get_samples('ECG_II')

plt.plot(raw_signals)
plt.show()

# start_time = 0
# end_time = len(raw_signals)/sampling_rate
# x = np.linspace(start_time*1e3, end_time*1e3, int(end_time*360.0))
# x0 = np.linspace(start_time*1e3, end_time*1e3, int(end_time*500.0)+1)
# x0 = x0[:len(raw_signals)]
# y = np.interp(x, x0, raw_signals)
# signals, params = biosppy.signals.tools.smoother(signal=y, kernel='boxzen', size=8)


