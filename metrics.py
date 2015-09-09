import numpy as np
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import roc_curve, auc

def meanAccuracy(y_true, y_pred, th = 0.5):
    y = (y_pred > 0.5)
    pos_idcs = (y_true == 1)
    neg_idcs = (y_true == 0)

    n_events = y_true.shape[1]
    pos_accuracy = np.zeros((n_events,), 'float32')
    neg_accuracy = np.zeros((n_events,), 'float32')
    accuracy = np.zeros((n_events, ), 'float32')

    for i in np.arange(n_events):
        pos_accuracy[i] = np.mean(y[pos_idcs[:, i], i])
        neg_accuracy[i] = 1 - np.mean(y[neg_idcs[:, i], i])
        accuracy[i] = np.mean(y[:, i] == y_true[:, i])

    return pos_accuracy.mean(), neg_accuracy.mean(), accuracy.mean()

def meanAUC(y_true, y_pred):
    n_events = y_pred.shape[1]
    scores = np.zeros((n_events,), 'float32')
    for i in np.arange(n_events):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        scores[i] = auc(fpr, tpr)

    return scores, scores.mean()

def kappa(y_true, y_pred):
    O = cm(y_true, y_pred)

    N = max(max(y_true), max(y_pred)) + 1
    W = np.zeros((N, N), 'float32')
    for i in np.arange(N):
        for j in np.arange(N):
            W[i, j] = (i - j) ** 2
    W /= ((N - 1) ** 2)

    hist_true = np.bincount(y_true, minlength = N)
    hist_pred = np.bincount(y_pred, minlength = N)
    E = np.outer(hist_true, hist_pred).astype('float32') / len(y_true)

    return 1 - (np.sum(W * O) / np.sum(W * E))

def confusion(y_true, y_pred):
    return cm(y_true, y_pred)

def ordinal_test(ys):
    pred = np.zeros((len(ys),), 'int32')
    for i, y in enumerate(ys):
        idx = -2
        for j in np.arange(len(y)):
            if y[j] < 0.5:
                idx = j - 1
                break
        if idx == -1:
            idx = 0
        if idx == -2:
            idx = len(y) - 1

        pred[i] = idx

    return pred
