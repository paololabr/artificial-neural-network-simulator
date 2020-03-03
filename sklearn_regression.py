

import sys
import numpy as np
import sklearn
import sklearn.metrics
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV

from neural_network import *

import matplotlib.pyplot as plt

from utility import *

'''
Ho semplicemente derivato la classe di MLPRegressor
'''

class SciKitNNRegr(MLPRegressor):
  def __init__(self, hidden_layer_sizes=(100, ), activation='relu', output_activation="identity", solver='adam', alpha=0.0001, batch_size='auto',
                       learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                       random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                       early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10,
                       max_fun=15000):
    super().__init__(hidden_layer_sizes, activation, solver, alpha, batch_size,
                       learning_rate, learning_rate_init, power_t, max_iter, shuffle,
                       random_state, tol, verbose, warm_start, momentum, nesterovs_momentum,
                       early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change,
                       max_fun)
    '''
        Nel caso volessimo inizializzare i pesi
        # attribute
        # coefs_list, length n_layers - 1,  The ith element in the list represents the weight matrix corresponding to layer i.
        # intercepts_list, length n_layers - 1, The ith element in the list represents the bias vector corresponding to layer i + 1.
    '''
    
def main():
    '''
    Leggo "cup/ML-CUP19-TR.csv" do in pasto il development test (75%) e valuto su test set rimanente.
    Risultati in MEE e MSE
    '''
    err, data, labels, testdata, testlabels = ReadData("cup/ML-CUP19-TR.csv", 0.75)

    if (err):
        print ("Error reading data")
    
    hidden_layer_sizes=(100, )
    activation='logistic'
    output_activation="identity" # non usata da MLPRegressor
    solver='adam'
    alpha=0.0001
    batch_size='auto'
    learning_rate='constant'
    learning_rate_init=0.001
    power_t=0.5
    max_iter=5000               # ben più grande del default
    shuffle=True
    random_state=None
    tol=0.0001
    verbose=False
    warm_start=False
    momentum=0.9
    nesterovs_momentum=True
    early_stopping=False
    validation_fraction=0.1
    beta_1=0.9
    beta_2=0.999
    epsilon=1e-08
    n_iter_no_change=10
    max_fun=15000
    
    nn = SciKitNNRegr(hidden_layer_sizes, activation, output_activation, solver, alpha, batch_size, learning_rate, learning_rate_init, power_t, max_iter, shuffle, random_state, tol, verbose, warm_start, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, max_fun)
    # nn = BaseNeuralNetwork ( hidden_layer_sizes=(2,), hidden_activation="threshold", momentum=0, alpha=0 )

    params={'alpha': [0.0001, 0.001, 0.01], '_etas': [0.05, 0.01], 'momentum': [0.3, 0.8]}
   
    ResList , minIdx = GridSearchCV(nn, params, data, labels, EuclideanLossFun, 2)

    print ('--- Res: ----')
    print (ResList)
    print ('-------------')
    print ('Best: ')
    print (ResList[minIdx])
    
    n_folds = 2

    nn.fit(data, labels)
    result = nn.predict(testdata)

    print("Test set MEE: " + str(AvgLoss(result, testlabels, EuclideanLossFun)) )
    print("Test set MSE: " + str(AvgLoss(result, testlabels, SquareLoss )))
    
main()