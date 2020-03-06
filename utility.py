import numpy as np
import csv
import os
import itertools
from itertools import product
from pathlib import Path
import random
   
def readMonk(filename, devfraction = 1, shuffle = False):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    resData = []
    try:
        with open(Path(dir_path + '/' + filename)) as infile:
            reader = csv.reader(infile, delimiter=" ")
            '''
            1. class: 0, 1 
            2. a1:    1, 2, 3
            3. a2:    1, 2, 3
            4. a3:    1, 2
            5. a4:    1, 2, 3
            6. a5:    1, 2, 3, 4
            7. a6:    1, 2
            8. Id:    (A unique symbol for each instance)
            '''
            for row in reader:
                label = int(row[1])

                rowdata = np.zeros(17)
                rowdata[int(row[2]) - 1] = 1
                rowdata[int(row[3]) + 2] = 1
                rowdata[int(row[4]) + 5] = 1
                rowdata[int(row[5]) + 7] = 1
                rowdata[int(row[6]) + 10] = 1
                rowdata[int(row[7]) + 14] = 1

                resData.append((rowdata, label))

            if (shuffle):
                random.shuffle(resData)

            if  0 < devfraction < 1:
                n = int(devfraction*len(resData))
                return list(zip(*resData[:n]))[0], list(zip(*resData[:n]))[1], list(zip(*resData[n:]))[0], list(zip(*resData[n:]))[1],

            return list(zip(*resData))[0], list(zip(*resData))[1], [], []

    except IOError:
        print('File ' + str(Path(dir_path)) + '/' + filename + ' not accessible')
        return [], [], [], []

def ReadData(filename, devfraction):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    try:
        with open(Path(dir_path + '/' + filename)) as infile:
            reader = csv.reader(infile, delimiter=",")
            labels = []
            data = []
            for row in reader:
                if row[0][0] != '#':
                    # Id, 20 data, 2 label
                    labels.append([float(row[21]), float(row[22])])
                    data.append([float(x) for x in row[1:21]])

        n = int(devfraction*len(data))

        return False, data[:n], labels[:n], data[n:], labels[n:]

    except IOError:
        print('File ' + str(Path(dir_path)) + '/' + filename + ' not accessible')
        return True, [], [], [], []

def AvgLoss(result, yorig, loss):
    ret = 0.
    for i, res in enumerate(result):
        ret += loss(res, yorig[i])

    return ret / len(result)

def SquareLoss(y, lb):
    return np.dot(np.array(y - lb).T, np.array(y - lb))

def EuclideanLossFun(y, z):
    dotval = np.dot(np.array(y - z).T, np.array(y - z))
    return np.sqrt(dotval)

def cross_val(model, data, labels, loss, folds=5):
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
    attribm = (dir(model))
    grid = GetParGrid(params, attribm)
    resList = []
    for p in grid:
        for k in p.keys():
            if k in attribm:
                setattr(model, k, p[k])
        res=cross_val(model,data,labels,loss,folds)
        resList.append([p, res])
    
    idx_min = np.argmin([it[1] for it in resList]).item()
    return resList, idx_min
           
def GetParGrid(params, attribm):
    res = []
    params = [params]
    for p in params:

        for key in [key for key in p if not key in attribm]:
            print ("warning: skipped param " + key + " (not found)")
            del p[key] 
       
        items = sorted(p.items())
        if items == []:
            return
        else:
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                res.append(params)
    return res
