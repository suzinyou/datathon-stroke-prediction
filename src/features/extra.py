import numpy as np 
import os
import pandas as pd

import biosppy
import joblib
from glob import glob
import sys

dir_path = '/home/jungkap/Documents/datathon/vitaldb/team3_final'
csv_files = glob('{}/*.csv'.format(dir_path))

prefixs = ['Solar 8000M/', 'Solar8000/']
feature_data = []
case_ids = []
for fpath in csv_files:
    record_id = int(fpath[-9:-4])
    df = pd.read_csv(fpath)
    if df[prefixs[0]+'HR'].notnull().sum() > df[prefixs[0]+'HR'].notnull().sum():
        p = prefixs[0]
    else:
        p = prefixs[1]

    feature = [record_id]
    col = 'BIS/BIS'
    
    mask = df[col].notnull()
    if not np.any(mask):
        feature.append( np.nan )
    else:
        feature.append( np.sum(df.loc[mask, col] <= 25) )
        
    col = '{}ART_SBP'.format(p)
    mask = df[col].notnull()
    if not np.any(mask):
        feature.append( np.sum(df.loc[mask, col] <= 80) )
    else:
        feature.append( np.sum(df.loc[mask, col] <= 25) )        

    col = '{}ART_MBP'.format(p)
    mask = df[col].notnull()
    if not np.any(mask):    
        feature.append( np.sum(df.loc[mask, col] <= 65) )
    else:
        feature.append( np.sum(df.loc[mask, col] <= 25) )                

    col = '{}ART_DBP'.format(p)
    mask = df[col].notnull()
    if not np.any(mask):
        feature.append( np.sum(df.loc[mask, col] >= 110) )    
    else:
        feature.append( np.sum(df.loc[mask, col] <= 25) )                

    col = '{}HR'.format(p)
    mask = df[col].notnull()
    if not np.any(mask):
        feature.append( np.sum(df.loc[mask, col] >= 100) )        
    else:
        feature.append( np.sum(df.loc[mask, col] <= 25) )                

    col = '{}HR'.format(p)
    mask = df[col].notnull()
    if not np.any(mask):    
        feature.append( np.sum(df.loc[mask, col] <= 60) )        
    else:
        feature.append( np.sum(df.loc[mask, col] <= 25) )                

    col = '{}PLETH_SPO2'.format(p)
    mask = df[col].notnull()
    if not np.any(mask):
        feature.append( np.sum(df.loc[mask, col] <= 90) )            
    else:
        feature.append( np.sum(df.loc[mask, col] <= 25) )        

    feature_data.append(feature)

    cols = ['CaseID', 'BIS/BIS <= 25',
     'ART_SBP <= 80',
     'ART_MBP <= 65',
     'ART_DBP >= 110',
     'HR >= 100',
     'HR <= 60',
     'PLETH_SPO2 <= 90']

    feature_df = pd.DataFrame(feature_data, columns=cols)
    #feature_df['CaseID'] = case_ids

    feature_df.to_csv('/home/jungkap/Documents/datathon/extra_feature.csv', index=False)




    


    df.shape