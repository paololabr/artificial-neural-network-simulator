
import sys
from neural_network import *
from functions import _euclidean_loss
from utility import ReadData
from utility import GridSearchCV
from utility import getRandomParams

def main():

    data, labels, _, _ = ReadData("cup/ML-CUP19-TR.csv", 0.90)
    
    if (len(data) == 0):
        print ("Error reading data")
        exit()

    nn = MLPRegressor(n_iter_no_change=10)
    
    params=[ {'hidden_layer_sizes': [(50,) ,(100,) ,(50,50)] , 'alpha': [0. , 0.001, 0.005, 0.01, 0.05], 'batch_size': [1 , 10 , 50 , 200, len(data)],
         'learning_rate': ['constant'], 'learning_rate_init': [0.01, 0.05, 0.1], 'momentum': [0., 0.1, 0.7, 0.9],
        'early_stopping': [True, False], 'activation': ['relu', 'tanh', 'logistic'] } ]

    '''
    params=[ {'hidden_layer_sizes': [(50,) ,(100,) ,(50,50)] , 'alpha': [0. , 0.001, 0.005, 0.01, 0.05], 'batch_size': [1 , 10 , 50 , 200, len(data)],
         'learning_rate': ['linear', 'adaptive'], 'learning_rate_init': [0.02, 0.1, 0.2], 'momentum': [0., 0.1, 0.7, 0.9],
        'early_stopping': [True, False], 'activation': ['relu', 'tanh', 'logistic'] } ]
    '''

    randparams = getRandomParams(params)
    
    ResList , minIdx = GridSearchCV(nn, randparams, data, labels, _euclidean_loss, 5, True)

main()
