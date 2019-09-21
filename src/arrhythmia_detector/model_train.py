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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import class_weight
import tensorflow as tf

input_size = 256

df = pd.read_pickle('/home/jungkap/Documents/datathon/qrs_data.pkl')
df = df.dropna(how='any', axis=0)

df['sample_weights'] = class_weight.compute_sample_weight('balanced', df['target'])

target_encoder = OneHotEncoder(sparse=False).fit(df['target'].values.reshape(-1, 1))
n_classes = df['target'].nunique()

# train/test set split
test_records = [104, 202, 214, 223, 231, 122]
test_mask = np.in1d(df['record_id'], test_records)
train_mask = ~test_mask

train_df = df[train_mask]
test_df = df[test_mask]

# downsampling for normal beats
normal_mask = train_df['target'] == 'NORMAL'
tr1 = train_df[normal_mask].sample(n=6000, replace=False)
tr2 = train_df[~normal_mask]
train_df = pd.concat((tr1, tr2))


target_encoder = OneHotEncoder().fit(df['target'].values.reshape(-1, 1))

X_train = train_df.iloc[:, :input_size].values.reshape([-1, input_size, 1])
X_train2 = train_df.iloc[:, -4:-1].values

y_train = target_encoder.transform( train_df.iloc[:, input_size].values.reshape(-1, 1) )
w_train = train_df.iloc[:, -1].values

X_test = test_df.iloc[:, :input_size].values.reshape([-1, input_size, 1])
X_test2 = test_df.iloc[:, -4:-1].values
y_test = target_encoder.transform( test_df.iloc[:, input_size].values.reshape(-1, 1) )

kernel_size = 16

cnn_inputs = Input(shape=(input_size, 1))
feature_inputs = Input(shape=(3,))

x = Conv1D(32, kernel_size=kernel_size, strides=4, input_shape=(input_size, 1))(cnn_inputs)
x = BatchNormalization()(x)
x = ELU()(x)

x = Conv1D(64, kernel_size=kernel_size, strides=1)(x)
x = BatchNormalization()(x)
x = ELU()(x)
x = MaxPooling1D(pool_size=2)(x)

# x = Conv1D(128, kernel_size=kernel_size, strides=2)(x)
# x = BatchNormalization()(x)
# x = ELU()(x)

# x = Conv1D(128, kernel_size=kernel_size, strides=1)(x)
# x = BatchNormalization()(x)
# x = ELU()(x)
# x = MaxPooling1D(pool_size=2)(x)
x = Flatten()(x)
x = Model(inputs=cnn_inputs, outputs=x)

y = Dense(4, activation="relu")(feature_inputs)
y = Model(inputs=feature_inputs, outputs=y)

# combine the output of the two branches
combined = concatenate([x.output, y.output])

z = Dropout(0.5)(combined)
z = Dense(n_classes, activation='softmax')(z)
 
# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=[x.input, y.input], outputs=z)

optimizer = optimizers.Adam()

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


model.fit(
    x=[X_train, X_train2], 
    y=y_train, 
    batch_size=256, 
    epochs=10, 
    verbose=1, 
    validation_data=([X_test, X_test2], y_test), 
    shuffle=True, 
    sample_weight=None)

model.save('/home/jungkap/Documents/datathon/model.h5')

# evaluation
y_prob = model.predict([X_test, X_test2])

y_pred = np.argmax(y_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

pr, rc, f, sup = precision_recall_fscore_support(y_true, y_pred)
classes = target_encoder.categories_[0]

label_true = classes[y_true]
label_pred = classes[y_pred]

conf_mat = confusion_matrix(label_true, label_pred, labels=classes)

conf_df = pd.DataFrame(conf_mat, columns=classes)
conf_df['label'] = classes
conf_df = conf_df.set_index('label')
print(conf_df)