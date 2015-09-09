import numpy as np
import sys
import os

result = sys.argv[1]
_, y = np.load(result)

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
_, name = os.path.split(result)
f = open(name + '.csv', 'w')
f.write(output)
