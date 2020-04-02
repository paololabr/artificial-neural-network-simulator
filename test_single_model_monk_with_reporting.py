
import sys
from neural_network import *
from functions import _classification_loss
from utility import readMonk
from utility import GridSearchCV
from utility import getRandomParams

def main():

    Xtrain, ytrain, _, _ = readMonk("monks/monks-1.train")
    Xtest, ytest, _, _ = readMonk("monks/monks-1.test")
    
    if len(Xtrain) == 0 or len(Xtest) == 0:
        print ("Error reading data")
        exit()

    nn = MLPClassifier( max_iter=500, n_iter_no_change=10 )

    # funzionanti da Paolo
    # params = {"hidden_layer_sizes": [6], "alpha": 0.00, "batch_size": 4, "activation": "tanh", "learning_rate": "constant", "learning_rate_init": 0.6, "power_t": 0.5, "momentum": 0.08}
    # funzionanti da Lucio
    # params = {"hidden_layer_sizes": [6], "alpha": 0.001, "batch_size": 4, "activation": "tanh", "learning_rate": "constant", "learning_rate_init": 0.6, "power_t": 0.5, "momentum": 0.08}
    params = {"hidden_layer_sizes": [6], "alpha": 0.0, "batch_size": 10, "activation": "tanh", "learning_rate": "constant", "learning_rate_init": 0.7, "momentum": 0.5}

    nn.set_params (**params)
    
    # print ("Xtrain.shape", Xtrain.shape)
    # print ("ytrain.shape", ytrain.shape)
    # print ("Xtest.shape", Xtest.shape)
    # print ("ytest.shape", ytest.shape)

    nn.enable_reporting ( Xtest, ytest, "monk-test", "classification" )
    nn.fit (Xtrain, ytrain)

    predicted = nn.predict (Xtest)
    loss = _classification_loss (ytest, predicted)

    print ("avg classification loss on validation:", loss)

main()
