
import sys
from neural_network import *
from functions import _euclidean_loss
from utility import ReadData
from utility import GridSearchCV

def main():

    data, labels, testdata, testlabels = ReadData("cup/ML-CUP19-TR.csv", 0.90)
    
    if (len(data) == 0):
        print ("Error reading data")

    nn = MLPRegressor(n_iter_no_change=10)
    
    params=[ {'hidden_layer_sizes': [(50,) ,(100,) ,(50,50)] , 'alpha': [0, 0.001, 0.005, 0.01, 0.05], 'batch_size': [1 , 10 , 50 , 200, 'auto'],
         'learning_rate': ['constant'], 'learning_rate_init': [0.02, 0.1, 0.2], 'momentum': [0., 0.05, 0.7, 0.9],
        'early_stopping': ['True', 'False'], 'activation': ['relu', 'tanh', 'logistic'] } ]

    ResList , minIdx = GridSearchCV(nn, params, data, labels, _euclidean_loss, 5)

main()