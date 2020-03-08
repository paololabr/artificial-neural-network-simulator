import numpy as np
import csv
import os
import itertools
from itertools import product
from pathlib import Path
import random
from matplotlib import pyplot as plt
from functions import *
import pickle
import pprint
from datetime import datetime
   
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

        return data[:n], labels[:n], data[n:], labels[n:]

    except IOError:
        print('File ' + str(Path(dir_path)) + '/' + filename + ' not accessible')
        return [], [], [], []

def AvgLoss(result, yorig, loss):
    return np.mean(loss(result, yorig))

def SquareLoss(y, lb):
    return np.sum((y-lb) * (y-lb), axis = y.ndim - 1)

def EuclideanLossFun(y, z):
    return np.sqrt(SquareLoss(y,z))

def ClassErrFun(y,z):
    return sum (1 for x,k in zip(y,z) if x!=k)
    
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

##########################
# GRID SEARCH FUNCTIONS  #
##########################
def GridSearchCV(model, params, data, labels, loss, folds=5):
    os.makedirs ("grid_reports", exist_ok=True)

    timestamp = datetime.today().isoformat().replace(':','_')
    filename = "grid_reports/" + model.__class__.__name__ + "_" + timestamp

    with open(filename + ".gsv", 'w', buffering=1) as outt:
        attribm = (dir(model))
        grid = GetParGrid(params, attribm)
        resList = []
        for p in grid:
            for k in p.keys():
                if k in attribm:
                    setattr(model, k, p[k])
            res=cross_val(model,data,labels,loss,folds)
            resList.append([p, res])
            pprint.pprint(p, outt,  width=8000, compact=True)
            pprint.pprint(res, outt)
        
        idx_min = np.argmin([it[1] for it in resList]).item()

        # todo variance
        print("*** Best ***", file=outt)
        print('Loss: ' + str(resList[idx_min][1]) + '\tVariance: ', file=outt )
        pprint.pprint(resList[idx_min], outt, width=8000, compact=True)

        return resList, idx_min
           
def GetParGrid(params, attribm):
    res = []
    params = [params]
    for line in params:
        for p in line:

            for key in [key for key in p if not key in attribm]:
                print ("warning: skipped param " + key + " (not found)")
                del p[key] 
        
            items = sorted(p.items())
            if items == []:
                return
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    par = dict(zip(keys, v))
                    res.append(par)
    return res

def readGridSearchFile(filename):
    val = 0.
    with open(Path(filename)) as infile:
        for line in infile:
            if line.startswith('Loss: '):
                res = line.split('\t')
                strs = res[0][6:]
                val = float(strs)
                return val
    
    return None

def getBestRes(fileprefix, directory):
    bestScore=None
    bestParams=None
    for f in os.listdir(directory):
        if f.startswith(fileprefix) and f.endswith(".gsv"):
            BestRes, _ = readGridSearchFile(f) 
            if bestScore==None or bestScore > BestRes[1]:
                bestScore = BestRes[1]
                bestParams = BestRes
            continue
        else:
            continue

    return bestParams    

##########################
#     PLOT FUNCTIONS     #
##########################
def CreateLossPlot(filename):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    train_loss = []
    valid_loss = []
    
    try:
        with open(Path(filename)) as infile:
            for line in infile:
                if line.startswith('# parameters:'):
                    param_line = line
                else:
                    ln = line.split('\t')
                    if (ln[0].isdigit()):
                        train_loss.append(float(ln[1]))
                        valid_loss.append(float(ln[2]))
        
        epoch_count = range(1, len(train_loss) + 1)
        plt.plot(epoch_count, train_loss, 'b-')
        plt.legend(['Training Loss', 'Test Loss'])

        plt.plot(epoch_count, valid_loss, 'r--')
        plt.legend(['Validation Loss', 'Test Loss'])

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(filename + '.png', bbox_inches='tight')

    except IOError:
        print('File ' + str(Path(dir_path)) + '/' + filename + ' not accessible')
        return True, [], [], [], []
