

import sys
import numpy as np
import sklearn 
import sklearn.metrics
from sklearn.neural_network import MLPRegressor as SKLRegressor
from sklearn.model_selection import train_test_split

from neural_network import MLPRegressor

from functions import _euclidean_loss
from utility import ReadData

class SciKitNNRegr(SKLRegressor):
    def __init__(self, hidden_layer_sizes=(100, ), activation = "relu", solver='sgd', alpha=0.0001, batch_size='auto',
                       learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                       random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                       early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10,
                       max_fun=15000):
        super().__init__(hidden_layer_sizes = hidden_layer_sizes, activation = activation, solver = solver, alpha = alpha, batch_size = batch_size,
                       learning_rate = learning_rate, learning_rate_init = learning_rate_init, power_t = power_t, max_iter= max_iter, shuffle = shuffle,
                       random_state = random_state, tol = tol, verbose = verbose, warm_start = warm_start, momentum = momentum, nesterovs_momentum = nesterovs_momentum,
                       early_stopping = early_stopping, validation_fraction = validation_fraction, beta_1 = beta_1, beta_2 = beta_2, epsilon = epsilon, n_iter_no_change = n_iter_no_change,
                       max_fun = max_fun)
        '''
        Nel caso volessimo inizializzare i pesi
        # attribute
        # coefs_list, length n_layers - 1,  The ith element in the list represents the weight matrix corresponding to layer i.
        # intercepts_list, length n_layers - 1, The ith element in the list represents the bias vector corresponding to layer i + 1.
        '''

def main():
    
    data, labels, _, _ = ReadData("cup/ML-CUP19-TR.csv", 0.90)

    if (len(data) == 0):
        print ("Error reading data")
        exit()

    Xtrain, Xtest, ytrain, ytest = train_test_split (data, labels, shuffle=True)

    constituent_params = [
        {'activation': 'tanh', 'alpha': 0.0028181084594129657, 'batch_size': 1, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'linear', 'learning_rate_init': 0.005887497817889539, 'momentum': 0.6415040512902583},
        {'activation': 'logistic', 'alpha': 0.0005951915361345095, 'batch_size': 1, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'adaptive', 'learning_rate_init': 0.054980020763198, 'momentum': 0.8050419419933234},
        {'activation': 'tanh', 'alpha': 0.0014347108668793685, 'batch_size': 5, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'linear', 'learning_rate_init': 0.03018084193934551, 'momentum': 0.7819029994042981},
        {'activation': 'tanh', 'alpha': 0.0007612380141853892, 'batch_size': 5, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'linear', 'learning_rate_init': 0.038179078019998106, 'momentum': 0.06888777870324973},
        {'activation': 'logistic', 'alpha': 0.0002188148237404819, 'batch_size': 1, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'constant', 'learning_rate_init': 0.016437178740396977, 'momentum': 0.483140617180181},
        {'activation': 'logistic', 'alpha': 0.0008463654654544185, 'batch_size': 1, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'adaptive', 'learning_rate_init': 0.07968904898581924, 'momentum': 0.2954139368874502},
        {'activation': 'logistic', 'alpha': 0.0009044205421256557, 'batch_size': 1, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'adaptive', 'learning_rate_init': 0.06205633095090332, 'momentum': 0.01510303444629194},
        {'activation': 'logistic', 'alpha': 0.0017164735753597117, 'batch_size': 5, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'linear', 'learning_rate_init': 0.051375660832639905, 'momentum': 0.5586182890198031},
        {'activation': 'logistic', 'alpha': 0.001542764264801877, 'batch_size': 5, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'constant', 'learning_rate_init': 0.020231597719749344, 'momentum': 0.5704950000170622},
        {'activation': 'logistic', 'alpha': 0.0016753169392290158, 'batch_size': 5, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'constant', 'learning_rate_init': 0.06569622348675933, 'momentum': 0.5544087332459164},
    ]    

    num_conf = len(constituent_params)
    if (len(sys.argv) > 1) and  (0 < int(sys.argv[1]) < len(constituent_params)):
        num_conf = int (sys.argv[1])
    
    for index, parameters_dict in enumerate(constituent_params):
        nn = SciKitNNRegr()
        nn.set_params (**parameters_dict)
        try:
            nn.fit(Xtrain, ytrain)

        except Exception as e:
            print("model " + str(index) + ": " + str(e))
        
        else:
            predicted = nn.predict (Xtest)
            lossv = _euclidean_loss (ytest, predicted)

            predicted = nn.predict (Xtrain)
            losst = _euclidean_loss (ytrain, predicted)

            print ("model " + str(index) + " MEE (train/validation): " + str(losst) + "/" + str(lossv))
            if (index == (num_conf-1)):
                break

    '''
    par = nn.get_params()
    params = {"activation": "tanh", "alpha": 0.0028181084594129657, "batch_size": 1, "early_stopping": False, "hidden_layer_sizes": [50, 50], "learning_rate": "linear", "learning_rate_init": 0.005887497817889539, "momentum": 0.6415040512902583}
    nn.set_params (**params)
    '''
    '''
    ResList , minIdx = GridSearchCV(nn, params, data, labels, EuclideanLossFun, 2)

    print ('--- Res: ----')
    print (ResList)
    print ('-------------')
    print ('Best: ')
    print (ResList[minIdx])
    
    n_folds = 2
    '''

    
    
main()