import numpy as np
from sklearn import svm
from metrics import AUC
import sys
import os

results_dir = './all_results/'  # This is the path where all test_valid results stored
results = ['resize3_c1r4p5_f9n128r35p1', 'len2560p1_resize3_bs_c1r4p5_f9n128r35p1', 
           'len4096_downsample4_resize3_bs_c7p7_f9n128_r35p1',
          ]  # This is the list of models that have been trained

combine_stride = 1
test_valid_interval = 10

num_results = len(results)
num_classes = 6
cross_valid = ['01', '23', '45', '67'] # Cross_valid subsets.

L, _ = np.load('eeg_train.npy')
eeg_shape = np.zeros([12, 8])
for i in xrange(12):
    for j in xrange(8):
        eeg_shape[i,j]=L[i,j].shape[0]

# get the location of subject 2's series 2
# we simply drop this series because the prediction result of it is far from precise
fake_s = (eeg_shape[:,0].sum() + 9)/test_valid_interval + (eeg_shape[1, 0] + 9)/test_valid_interval
fake_e = (eeg_shape[:,0].sum() + 9)/test_valid_interval + (eeg_shape[1,:2].sum() + 9)/test_valid_interval


labels = []
data = []
for i in np.arange(num_classes):
    data.append([])
for i in np.arange(num_results):
    print results[i]
    l = np.zeros((0, num_classes), 'int32')
    d = np.zeros((0, num_classes), 'int32')
    for j in np.arange(len(cross_valid)):
        temp_l, temp_d = np.load(os.path.join(results_dir,
                                              'test_valid_' + results[i] + '_v' + cross_valid[j] + '.npy'))

        if j == 0: # Drop subject 2's series 2
            temp_l = np.concatenate((temp_l[:fake_s, :], temp_l[fake_e:, :]), axis = 0)
            temp_d = np.concatenate((temp_d[:fake_s, :], temp_d[fake_e:, :]), axis = 0)

        l = np.concatenate((l, temp_l[::combine_stride, :]), axis = 0)
        d = np.concatenate((d, temp_d[::combine_stride, :]), axis = 0)
        
    if i == 0:
        labels = l
        for j in np.arange(num_classes):
            data[j] = d[:, j].reshape((-1, 1))
    else:
        for j in np.arange(num_classes):
            data[j] = np.concatenate((data[j], d[:, j].reshape((-1, 1))), axis = 1)

np.save('valid_combine.npy', [labels, data, results])

