
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

    # funzionanti da Paolo (curva stabile, alpha>0 e n_iter_no_change alto)
    # params={"hidden_layer_sizes": [6], "alpha": 0.003, "activation": "tanh", "learning_rate": "constant", "learning_rate_init": 0.82,  "momentum": 0.6, "n_iter_no_change": 80}
    params={"hidden_layer_sizes": [6], "alpha": 0.003, "activation": "tanh", "learning_rate": "constant", "learning_rate_init": 0.82,  "momentum": 0.6, "n_iter_no_change": 80}
    
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
