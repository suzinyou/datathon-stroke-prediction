import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

input_size = 256

df = pd.read_pickle('/home/jungkap/Documents/datathon/qrs_data.pkl')
df = df.dropna(how='any', axis=0)


ii = np.where((df['target'] == 'PVC') & (df['record_id'] == 201))[0]


for j in np.random.choice(ii, size=5, replace=False):
    plt.plot(df.iloc[j, :256])

plt.show()
