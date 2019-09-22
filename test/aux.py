import numpy as np
import os
import sys

flist = np.loadtxt('/home/jungkap/Documents/datathon/vitaldb/file.list', dtype=int)
len(flist)



for record_id in flist:
    fname = '{id:05d}.vital'.format(id=record_id)
    fpath = os.path.join('/home/jungkap/Documents/datathon/vitaldb', fname)

    if not os.path.exists(fpath):
        print('get {}'.format(fname))


    
# for ipath in vital_files:
#     record_id = int(ipath[-11:-6])
#     if record_id not in flist:
#         print(record_id)
    