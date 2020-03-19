
from neural_network import *
from utility import ReadData
from ensambler import Ensambler
from functions import _euclidean_loss

def main():

    data, labels, _, _ = ReadData("cup/ML-CUP19-TR.csv", 0.90)
    
    if (len(data) == 0):
        print ("Error reading data")
        exit()

    n1 = MLPRegressor(
        n_iter_no_change=10, max_iter=5,
        hidden_layer_sizes=[50, 50], alpha=0.001, batch_size=10, activation="logistic", learning_rate="constant", learning_rate_init=0.02, early_stopping=True
    )
    n2 = MLPRegressor(
        n_iter_no_change=10, max_iter=5,
        hidden_layer_sizes=[100,], alpha=0.001, batch_size=10, activation="logistic", learning_rate="constant", learning_rate_init=0.02, early_stopping=True
    )
    n3 = MLPRegressor(
        n_iter_no_change=10, max_iter=5,
        hidden_layer_sizes=[200,], alpha=0.001, batch_size=50, activation="logistic", learning_rate="constant", learning_rate_init=0.02, early_stopping=True
    )
    
    Xtrain, Xtest, ytrain, ytest = train_test_split (data, labels, shuffle=True)
    # print ("data.shape", data.shape)
    # print ("labels.shape", labels.shape)
    # print ("Xtrain.shape", Xtrain.shape)
    # print ("ytrain.shape", ytrain.shape)
    # print ("Xtest.shape", Xtest.shape)
    # print ("ytest.shape", ytest.shape)

    ens = Ensambler ([n1, n2, n3])
    ens.enable_reporting (Xtest, ytest, "hold_out_validation_set", accuracy="euclidean")
    ens.fit (Xtrain, ytrain)
    predicted = ens.predict (Xtest)
    loss = _euclidean_loss (ytest, predicted)

    print ("avg ensemble euclidean loss on validation:", loss)

    ens.write_constituent_vs_ensamble_report (Xtest, ytest, dataset_name="hold_out_validation_set")

main()
