import numpy as np
from sklearn import svm
from metrics import AUC
import sys
import os
import time

labels, data, results = np.load('valid_combine.npy')
num_results = len(results)
num_classes = 6

auc = np.zeros((num_results, num_classes), 'float32')
candidate_idcs = []
for i in np.arange(num_classes):
    candidate_idcs.append(list(xrange(num_results)))
selected_idcs = [[], [], [], [], [], []]
selected_features = [[], [], [], [], [], []]

selected_sum = []
for i in np.arange(num_classes):
    selected_sum.append(0)

print num_results
for i in np.arange(num_results):
    print i
    st = time.time()
    sub_auc = [[], [], [], [], [], []]

    if i > 0:
        for j in np.arange(num_classes):
            selected_sum[j] = selected_sum[j] + data[j][:, selected_idcs[j][-1]]
    
    for k in np.arange(num_classes):
        s = time.time()
        for idx in candidate_idcs[k]:
            sub_auc[k].append(AUC(labels[:, k], selected_sum[k] + data[k][:, idx]))
        e = time.time()
        print (e - s)
        max_idx = np.argmax(np.asarray(sub_auc[k]))
        selected_idcs[k].append(candidate_idcs[k][max_idx])
        candidate_idcs[k].remove(candidate_idcs[k][max_idx])
        auc[i, k] = sub_auc[k][max_idx]
        selected_features[k].append(results[selected_idcs[k][-1]])
        
    et = time.time()
    print "elapsed time is %f seconds, auc is: " % (et - st)
    print auc[i]
    np.save('selection_result.npy', [auc, selected_idcs, selected_features])
