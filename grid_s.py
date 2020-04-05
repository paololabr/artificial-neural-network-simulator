
import sys
from neural_network import *
from functions import _euclidean_loss
from utility import ReadData
from utility import GridSearchCV
from utility import getRandomParams

import tqdm

def main():

    n_configurations = 10
    if len(sys.argv) > 1:
        n_configurations = int (sys.argv[1])

    data, labels, _, _ = ReadData("cup/ML-CUP19-TR.csv", 0.90)
    
    if (len(data) == 0):
        print ("Error reading data")
        exit()

    # nn = MLPRegressor(n_iter_no_change=10, max_iter=5)
    nn = MLPRegressor(n_iter_no_change=10, max_iter=500)
    
    params=[ {'hidden_layer_sizes': [(10,10), (20,), (50,) ,(100,), (50,50)] , 'alpha': [0., 0.05], 
        'batch_size': [1, 5, 10, 50, 100, 'auto', 500, len(data)],
         'learning_rate': ['constant', 'adaptive', 'linear'], 'learning_rate_init': [0.001, 0.1], 'momentum': [0., 0.9],
        'early_stopping': [True, False], 'activation': ['relu', 'tanh', 'logistic'],
        "weights_init_fun": ["random_uniform", "random_normal"], "weights_init_value": [0.2, 0.8] } ]

    for i in tqdm.tqdm (range (n_configurations), desc="configurations"):
        randparams = getRandomParams(params)
        ResList, minIdx = GridSearchCV(nn, randparams, data, labels, _euclidean_loss, 5, uniquefile=True, write_best=False)

main()
