
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

    nn = MLPRegressor(
        n_iter_no_change=10, max_iter=500,
    )

    params = {'activation': 'logistic', 'alpha': 0.0006, 'batch_size': 1, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'adaptive', 'learning_rate_init': 0.055, 'momentum': 0.8050419419933234}
    #params = {'activation': 'logistic', 'alpha': 0.0006, 'batch_size': 1, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'adaptive', 'learning_rate_init': 0.055, 'momentum': 0.8050419419933234, "weights_init_fun": "random_uniform", "weights_init_value":0.35}
    #params={"hidden_layer_sizes": [15,15], "alpha": 0.01, "activation": "tanh", "learning_rate": "constant", "learning_rate_init": 0.8,  "momentum": 0.8, "n_iter_no_change": 10, "weights_init_fun": "random_normal", "weights_init_value":0.8}
    #params={"hidden_layer_sizes": [15,15], "alpha": 0.001,'batch_size': 8, "activation": "tanh", "learning_rate": "constant", "learning_rate_init": 0.8,  "momentum": 0.6, "n_iter_no_change": 10, "weights_init_fun": "random_normal", "weights_init_value":0.7}
    nn.set_params (**params)
    
    Xtrain, Xtest, ytrain, ytest = train_test_split (data, labels, shuffle=True)
    # print ("data.shape", data.shape)
    # print ("labels.shape", labels.shape)
    # print ("Xtrain.shape", Xtrain.shape)
    # print ("ytrain.shape", ytrain.shape)
    # print ("Xtest.shape", Xtest.shape)
    # print ("ytest.shape", ytest.shape)

    nn.enable_reporting ( Xtest, ytest, "hold_out_validation", "euclidean" )
    nn.fit (Xtrain, ytrain)

    predicted = nn.predict (Xtest)
    loss = _euclidean_loss (ytest, predicted)

    print ("avg euclidean loss on validation:", loss)

main()
