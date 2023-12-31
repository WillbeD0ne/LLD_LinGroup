import numpy as np
from sklearn import metrics

def ACC(output, target):
    if len(output.shape) == 2:
        y_pred = output.argmax(1)
    else:
        y_pred = output
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.accuracy_score(y_true, y_pred)

def Cohen_Kappa(output, target):
    if len(output.shape) == 2:
        y_pred = output.argmax(1)
    else:
        y_pred = output
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.cohen_kappa_score(y_true, y_pred)

def F1_score(output, target):
    if len(output.shape) == 2:
        y_pred = output.argmax(1)
    else:
        y_pred = output
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.f1_score(y_true, y_pred, average='macro')

def Recall(output, target):
    if len(output.shape) == 2:
        y_pred = output.argmax(1)
    else:
        y_pred = output
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.recall_score(y_true, y_pred, average='macro')

def Precision(output, target):
    if len(output.shape) == 2:
        y_pred = output.argmax(1)
    else:
        y_pred = output
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.precision_score(y_true, y_pred, average='macro')

def cls_report(output, target):
    if len(output.shape) == 2:
        y_pred = output.argmax(1)
    else:
        y_pred = output
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.classification_report(y_true, y_pred, digits=4)


def confusion_matrix_(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.confusion_matrix(y_true, y_pred)