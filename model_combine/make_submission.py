import numpy as np
import sys
import os
import time

results_dir = './all_test_results'

auc, _, results = np.load('selection_result.npy')
auc = np.asarray(auc)
num_classes = 6
for i in np.arange(num_classes):
    idx = np.argmax(auc[:, i])
    results[i] = results[i][:idx + 1]
    print len(results[i])

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


y = np.asarray(data_noncv).T

first_line = 'id,HandStart,FirstDigitTouch,BothStartLoadPhase,LiftOff,Replace,BothReleased'
_, labels = np.load('eeg_test.npy')
subs = np.arange(1, 13)
test_series = [9, 10]
n_events = 6
csv = []
c = 0
for sub in subs:
    sub_csv = []
    for series in test_series:
        series_csv = []
        l = labels[sub - subs[0], series - test_series[0]]
        for k in np.arange(len(l)):
            row = []
            row.append('subj' + str(sub) + '_series' + str(series) + '_' + str(k))
            for n in np.arange(n_events):
                row.append(str(y[c, n]))
            row = ','.join(row)
            series_csv.append(row)
            c += 1
        sub_csv.append(series_csv)
    csv.append(sub_csv)

output = []
output.append(first_line)
for series in test_series:
    for sub in subs:
        output = output + csv[sub - subs[0]][series - test_series[0]]

output = '\n'.join(output)
f = open('submission.csv', 'w')
f.write(output)
