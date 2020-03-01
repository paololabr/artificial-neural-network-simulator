import numpy as np
import csv
import os
import itertools
from itertools import product
   
def ReadCup(devfraction):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cuptrain = os.path.join(dir_path, "cup/ML-CUP19-TR.csv")

    with open(cuptrain) as infile:
        reader = csv.reader(infile, delimiter=",")
        labels = []
        data = []
        for row in reader:
            if row[0][0] != '#':
                # Id, 20 data, 2 label
                labels.append([float(row[21]), float(row[22])])
                data.append([float(x) for x in row[1:21]])

    n = int(devfraction*len(data))

    return data[:n], labels[:n], data[n:], labels[n:]

def AvgLoss(result, yorig, loss):
    ret = 0.
    for i, res in enumerate(result):
        ret += loss(res, yorig[i])

    return ret / len(result)

def SquareLoss(y, lb):
    return np.dot(y - lb, y - lb)

def EuclideanLossFun(y, z):
    return np.sqrt(np.dot(y - z, y - z))

def cross_val(model, data, labels, loss, folds=5):
    print(dir(model))
    X_tr_folds = np.array_split(data, folds)
    y_tr_folds = np.array_split(labels, folds)
    sumAvg = 0

    for i in range(folds):
        tr_data, test_data = np.concatenate(X_tr_folds[:i] + X_tr_folds[i+1:]), X_tr_folds[i]
        tr_labels, testlabels = np.concatenate(y_tr_folds[:i] + y_tr_folds[i+1:]), y_tr_folds[i]
        model.fit(tr_data, tr_labels)
        result = model.predict(test_data)
        sumAvg+= AvgLoss(result, testlabels, loss)

    return sumAvg / folds

def GridSearchCV(model, params, data, labels, loss, folds=5):
    grid = GetParGrid(params)
    resList = []
    attribm = (dir(model))
    for p in grid:
        for k in p.keys():
            if not k in attribm:
                print ('Not found')
            else:
                setattr(model, k, p[k])
        res=cross_val(model,data,labels,loss,2)
        resList.append(res)
    return resList
           
def GetParGrid(params):
    res = []
    params = [params]
    for p in params:
        items = sorted(p.items())
        if items == []:
            return
        else:
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                res.append(params)
    return res