import numpy as np 
import os
import pandas as pd
import joblib
from glob import glob



filepath = '/home/jungkap/Documents/datathon/arrhythmia_feature.pkl'
features = joblib.load(filepath)

for i in range(1, 6):
    filepath = '/home/jungkap/Documents/datathon/arrhythmia_feature{}.pkl'.format(i)
    tmp = joblib.load(filepath)
    for k, v in tmp.items():
        features[k] = v

cols = ['CaseID', 'AF_rhythm', 'AFIB_rhythm', 'NORMAL_rhythm', 'APC_beat', 'LBB_beat', 'NORMAL_beat', 'PAB_beat', 'PVC_beat', 'RBB_beat']
df = pd.DataFrame(features).T 
df = df.reset_index()
df.columns=cols

extra_df = pd.read_csv('/home/jungkap/Documents/datathon/extra_feature.csv')

merged_df = extra_df.merge(df, how='left', on='CaseID')
merged_df.loc[:, cols] = merged_df.loc[:, cols].fillna(0)


merged_df.to_csv('/home/jungkap/Documents/datathon/vital_signal_feature.csv', index=False)