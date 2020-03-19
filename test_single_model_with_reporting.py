
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
 #       hidden_layer_sizes=[50, 50], alpha=0.001, batch_size=10, activation="logistic", learning_rate="constant", learning_rate_init=0.02, early_stopping=True
    )
    
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
