import numpy as np
import os
import pandas as pd

csvdir = 'data/train'
n_subs = 12
n_series = 8
n_channels = 32

data = []
label = []

for sub in np.arange(n_subs):
    sub_data = []
    sub_label = []
    for series in np.arange(n_series):
        csv = 'subj' + str(sub + 1) + '_series' + str(series + 1) + '_data.csv'
        series_data = pd.read_csv(os.path.join(csvdir, csv))
        ch_names = list(series_data.columns[1:])
        series_data = np.array(series_data[ch_names], 'float32')
        sub_data.append(series_data)

        csv = 'subj' + str(sub + 1) + '_series' + str(series + 1) + '_events.csv'
        series_label = pd.read_csv(os.path.join(csvdir, csv))
        ch_names = list(series_label.columns[1:])
        series_label = np.array(series_label[ch_names], 'float32')
        sub_label.append(series_label)

    data.append(sub_data)
    label.append(sub_label)

np.save('eeg_train.npy', [data, label])
