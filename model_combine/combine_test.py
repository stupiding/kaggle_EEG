import numpy as np
import sys
import os
import time

results_dir = './all_results'
if len(sys.argv) != 2:
    sys.exit('Usage: combine_test.py <ff_result_file>')

auc, _, results = np.load(sys.argv[1])
auc = np.asarray(auc)
num_classes = 6
for i in np.arange(num_classes):
    idx = np.argmax(auc[:, i])
    results[i] = results[i][:idx + 1]
    print len(results[i])

# Combine the test result of cross valid models
# In our final submition, this is not used
"""
subsets = ['01', '23', '45', '67']
data_cv = []
for k in np.arange(num_classes):
    labels = []
    data = []
    count = 0
    for i in np.arange(len(results[k])):
        st = time.time()
        for j in np.arange(len(subsets)):
            l, d = np.load(os.path.join(results_dir, 'test_' + results[k][i] + '_v' + subsets[j] + '.npy'))

            if count == 0:
                labels = l
                data = d[:, k]
            else:
                data = data + d[:, k]
            count = count + 1
        et = time.time()
        print "%d, %d, elapsed time is %f seconds" % (k, i, et - st)
    data_cv.append(data / count)
data_cv = np.asarray(data_cv).T
"""

data_noncv = []
for k in np.arange(num_classes):
    labels = []
    data = []
    count = 0
    for i in np.arange(len(results[k])):
        st = time.time()
        l, d = np.load(os.path.join(results_dir, 'test_' + results[k][i] + '.npy'))

        if count == 0:
            labels = l
            data = d[:, k]
        else:
            data = data + d[:, k]
        count = count + 1
        et = time.time()
        print "%d, %d, elapsed time is %f seconds" % (k, i, et - st)
    data_noncv.append(data / count)
data_noncv = np.asarray(data_noncv).T

np.save('submit_test.npy', [labels, data_noncv])
