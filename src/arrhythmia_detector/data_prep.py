import wfdb
import numpy as np
from glob import glob
import biosppy
import os 
import cv2
import pandas as pd 

from matplotlib import pyplot as plt

mitdb_path = '/home/jungkap/Documents/datathon/alpha.physionet.org/files/mitdb/1.0.0/'
paths = glob('{}*.atr'.format(mitdb_path))
paths = [path[:-4] for path in paths]


symbol_abbr = ['APC', 'NORMAL', 'LBB', 'PVC', 'PAB', 'RBB']
symbol = ['A', 'N', 'L', 'V', '/', 'R']
symbol_map = {k: v for k, v in zip(symbol, symbol_abbr)}


# refer to ecg_annotator
PQ_DURATION = 0.20  # max PQ duration
QT_DURATION = 0.48  # max QT duration

sampling_rate = 360 # MIT-BH db

d1 = int(np.ceil(PQ_DURATION*sampling_rate)) + 4
d2 = int(np.ceil(QT_DURATION*sampling_rate)) + 7

qrs_data = []
qrs_label = []
qrs_rid = []
features = []
qrs_sub = []

for fid, record_name in enumerate(paths):
    rid = int(record_name[-3:])
    
    raw_signals, fields = wfdb.rdsamp(record_name, channels=[0]) 
    
    signals, params = biosppy.signals.tools.smoother(
        signal=raw_signals.squeeze(), kernel='boxzen', size=8)

    ann = wfdb.rdann(record_name, 'atr')
    rep_rr_dist = np.median(ann.sample[1:] - ann.sample[:-1])
    mask = np.in1d(ann.symbol, symbol)

    prev_index = ann.sample[0]
    for i in range(1, len(ann.symbol)-1):
        label, index = ann.symbol[i], ann.sample[i]
        aux_label = ann.aux_note[i]

        st = index - d1
        ed = index + d2
        if mask[i] and st >= 0 and ed < len(signals):
            y = signals[st:ed]
            rep_height = np.median(y)
            rr_dist1 = (index - prev_index)/rep_rr_dist
            rr_dist2 = (ann.sample[i+1] - index)/rep_rr_dist            

            peak_ratio = np.abs(rep_height - np.min(y)) / np.abs(np.max(y) - rep_height)

            qrs_data.append(y)
            qrs_label.append(label)
            qrs_sub.append(aux_label)
            qrs_rid.append(rid)
            features.append([rr_dist1, rr_dist2, peak_ratio])
        prev_index = index

tmp_df = pd.DataFrame(features, columns=["prev_rr", "next_rr", "peak_ratio"])
qrs_df = pd.DataFrame(qrs_data)
qrs_df['target'] = pd.Series([symbol_map[k] for k in qrs_label])
qrs_df['record_id'] = qrs_rid
qrs_df['target2'] = qrs_sub
qrs_df = pd.concat((qrs_df, tmp_df), axis=1)

qrs_df.to_pickle('/home/jungkap/Documents/datathon/qrs_data.pkl')

# visualization
# for label in symbol_abbr:
#     mask = qrs_df['target'] == label
#     t1 = qrs_df.loc[mask, ["prev_rr", "peak_ratio"]].sample(100)
#     plt.scatter(t1.iloc[:, 0], t1.iloc[:, 1], marker='.')
# plt.legend(symbol_abbr)
# plt.show()

stats_df = pd.DataFrame(
            np.zeros((48, len(symbol_abbr))),
            index=qrs_df['record_id'].unique(), 
            columns=symbol_abbr,
            dtype=int)

for group, sub_df in qrs_df.groupby('record_id'):
    stats = sub_df['target'].value_counts()
    for k, v in stats.iteritems():
        stats_df.loc[group, k] = v

    
########################## baseline classifier ##############################
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import class_weight

df = pd.read_pickle('/home/jungkap/Documents/datathon/qrs_data.pkl')
df = df.dropna(how='any', axis=0)


n_classes = df['target'].nunique()

input_size = 256
# train/test set split
#test_records = [104, 202, 214, 223, 231]
test_records = [104, 202, 214, 223, 231, 122]
test_mask = np.in1d(df['record_id'], test_records)
train_mask = ~test_mask

train_df = df[train_mask]
test_df = df[test_mask]



normal_mask = train_df['target'] == 'NORMAL'
tr1 = train_df[normal_mask].sample(n=10000, replace=False)
tr2 = train_df[~normal_mask]
train_df = pd.concat((tr1, tr2))


target_encoder = LabelEncoder().fit(df['target'].values.reshape(-1, 1))

# X_train = train_df.iloc[:, :input_size]
# y_train = target_encoder.transform( train_df.iloc[:, input_size].values.reshape(-1, 1) )
# w_train = train_df.iloc[:, -1].values

# X_test = test_df.iloc[:, :input_size]
# y_test = target_encoder.transform( test_df.iloc[:, input_size].values.reshape(-1, 1) )

col = train_df.columns.difference(['record_id', 'target'])
y_train = target_encoder.transform(train_df['target'])
y_test = target_encoder.transform(test_df['target'])

#model = LogisticRegression()
model = LGBMClassifier(n_estimators=30)
model.fit(train_df[col], y_train)

y_pred = model.predict(test_df[col])
y_true = target_encoder.inverse_transform( y_test )

symbol_abbr = ['APC', 'NORMAL', 'LBB', 'PVC', 'PAB', 'RBB']

accuracy_score(y_test, y_pred)
#jj = np.where(y_true == 'PVC')[0]

y_pred = target_encoder.inverse_transform( y_pred )

pr, rc, f, support = precision_recall_fscore_support(y_true, y_pred, labels=symbol_abbr)

conf_mat = confusion_matrix(y_true, y_pred, labels=symbol_abbr)

conf_df = pd.DataFrame(conf_mat, columns=symbol_abbr)
conf_df['label'] = symbol_abbr
conf_df = conf_df.set_index('label')
conf_df.to_csv('/home/jungkap/Documents/datathon/conf_mat2.csv')
