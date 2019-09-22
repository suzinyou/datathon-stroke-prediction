import pandas as pd
import numpy as np
#from ecg_model import ArrhymthmiaDetector, IMG_SIZE

from keras import optimizers
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential, Model
from keras.layers import (
    Conv1D, MaxPooling1D, Input,
    Dense, ELU, BatchNormalization, 
    Flatten, Dropout, ReLU, concatenate
)
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import class_weight
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


df = pd.read_pickle('/home/jungkap/Documents/datathon/afib_data.pkl')
#df = df.dropna(how='any', axis=0)
#df['sample_weights'] = class_weight.compute_sample_weight('balanced', df['target'])

input_size = 1250

target_encoder = OneHotEncoder(sparse=False).fit(df['target'].values.reshape(-1, 1))
n_classes = df['target'].nunique()

# train/test set split
by_patients = False

# 1) by patients
if by_patients:
    test_records = np.random.choice(23, size=3)
    #test_records = [0, 9, 22]
    test_mask = np.in1d(df['record_id'], test_records)
    # df.loc[test_mask, 'target'].value_counts()
    train_mask = ~test_mask

    train_df = df[train_mask]
    test_df = df[test_mask]
    train_df, val_df = train_test_split(train_df, test_size=20000, 
        random_state=928, shuffle=True, stratify=train_df['target'])
else:
    train_df, test_df = train_test_split(df, test_size=15000,
        random_state=928, shuffle=True, stratify=df['target'])
    train_df, val_df = train_test_split(train_df, test_size=15000, 
        random_state=928, shuffle=True, stratify=train_df['target'])        



# downsampling for normal beats
train_df['target'].value_counts()

tr1 = train_df[train_df['target'] == 'NORMAL'].sample(n=20000, replace=False)
tr2 = train_df[train_df['target'] == 'AFIB'].sample(n=20000, replace=False)
tr3 = train_df[train_df['target'] == 'AF']
train_df = pd.concat((tr1, tr2, tr3))
del tr1, tr2, tr3

target_encoder = OneHotEncoder().fit(df['target'].values.reshape(-1, 1))

X_train = train_df.iloc[:, :input_size].values.reshape([-1, input_size, 1])
#X_train2 = train_df.iloc[:, -4:-1].values

y_train = target_encoder.transform( train_df['target'].values.reshape(-1, 1) )
#w_train = train_df.iloc[:, -1].values

X_test = test_df.iloc[:, :input_size].values.reshape([-1, input_size, 1])
#X_test2 = test_df.iloc[:, -4:-1].values
y_test = target_encoder.transform( test_df['target'].values.reshape(-1, 1) )

X_val = val_df.iloc[:, :input_size].values.reshape([-1, input_size, 1])
y_val = target_encoder.transform( val_df['target'].values.reshape(-1, 1) )

#kernel_size = 32
kernel_size = 256

cnn_inputs = Input(shape=(input_size, 1))
feature_inputs = Input(shape=(3,))

x = Conv1D(64, kernel_size=kernel_size, strides=16, input_shape=(input_size, 1))(cnn_inputs)
x = BatchNormalization()(x)
x = ELU()(x)

x = Conv1D(128, kernel_size=32, strides=4)(x)
x = BatchNormalization()(x)
x = ELU()(x)
x = MaxPooling1D(pool_size=2)(x)

x = Flatten()(x)
x = Dense(256)(x)
x = Dropout(0.5)(x)

x = Dense(n_classes, activation='softmax')(x)
# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=cnn_inputs, outputs=x)

optimizer = optimizers.Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

history = model.fit(
    x=X_train, 
    y=y_train, 
    batch_size=512, 
    epochs=150, 
    verbose=1, 
    validation_data=(X_val, y_val), 
    shuffle=True, 
    sample_weight=None,
    callbacks=[es])


X_all = df.iloc[:, :input_size].values.reshape([-1, input_size, 1])
y_all = target_encoder.transform( df['target'].values.reshape(-1, 1) )

history = model.fit(
    x=X_all, 
    y=y_all, 
    batch_size=1024, 
    epochs=2, 
    verbose=1, 
    validation_data=(X_val, y_val), 
    shuffle=True, 
    sample_weight=None,
    callbacks=[es])


n = len(history.history['val_loss'])

from matplotlib import pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

model.save('/home/jungkap/Documents/datathon/afib_classifier.h5')


# evaluation

classes = target_encoder.categories_[0]

y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# accuracy_score(y_true, y_pred)

pr, rc, f, sup = precision_recall_fscore_support(y_true, y_pred)

label_true = classes[y_true]
label_pred = classes[y_pred]

conf_mat = confusion_matrix(label_true, label_pred, labels=classes)

conf_df = pd.DataFrame(conf_mat, columns=classes)
conf_df['label'] = classes
conf_df = conf_df.set_index('label')
print(conf_df)





