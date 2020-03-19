
from neural_network import *
from utility import ReadData
from ensambler import Ensambler
from functions import _euclidean_loss

def main():

    data, labels, _, _ = ReadData("cup/ML-CUP19-TR.csv", 0.90)
    
    if (len(data) == 0):
        print ("Error reading data")
        exit()

    constituent_params = [ 
        {'activation': 'logistic', 'alpha': 0.0005951915361345095, 'batch_size': 1, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'adaptive', 'learning_rate_init': 0.054980020763198, 'momentum': 0.8050419419933234},
        {'activation': 'logistic', 'alpha': 0.0002188148237404819, 'batch_size': 1, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'constant', 'learning_rate_init': 0.016437178740396977, 'momentum': 0.483140617180181},
        {'activation': 'logistic', 'alpha': 0.0017164735753597117, 'batch_size': 5, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'linear', 'learning_rate_init': 0.051375660832639905, 'momentum': 0.5586182890198031},
        {'activation': 'logistic', 'alpha': 0.001542764264801877, 'batch_size': 5, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'constant', 'learning_rate_init': 0.020231597719749344, 'momentum': 0.5704950000170622},
        {'activation': 'logistic', 'alpha': 0.0016753169392290158, 'batch_size': 5, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'constant', 'learning_rate_init': 0.06569622348675933, 'momentum': 0.5544087332459164},
        {'activation': 'logistic', 'alpha': 0.007614556427082947, 'batch_size': 1, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'linear', 'learning_rate_init': 0.026343499036647663, 'momentum': 0.7312710943570491},
        {'activation': 'tanh', 'alpha': 0.005400073011879714, 'batch_size': 5, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'linear', 'learning_rate_init': 0.05239219294784031, 'momentum': 0.4824596839921277},
        {'activation': 'logistic', 'alpha': 0.009112993997484475, 'batch_size': 1, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'adaptive', 'learning_rate_init': 0.03862901305077766, 'momentum': 0.071787162708924},
        {'activation': 'logistic', 'alpha': 0.0034215552021840904, 'batch_size': 10, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'linear', 'learning_rate_init': 0.09146062940899181, 'momentum': 0.4106370446565398},
        {'activation': 'logistic', 'alpha': 0.004236037776064028, 'batch_size': 5, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'linear', 'learning_rate_init': 0.08941448141555948, 'momentum': 0.8857394330819264}
    ]

    models = []
    for parameters_dict in constituent_params:
        nn = MLPRegressor ()
        nn.set_params (**parameters_dict)
        models.append (nn)
    
    Xtrain, Xtest, ytrain, ytest = train_test_split (data, labels, shuffle=True)
    # print ("data.shape", data.shape)
    # print ("labels.shape", labels.shape)
    # print ("Xtrain.shape", Xtrain.shape)
    # print ("ytrain.shape", ytrain.shape)
    # print ("Xtest.shape", Xtest.shape)
    # print ("ytest.shape", ytest.shape)

    ens = Ensambler (models, verbose=True)
    ens.enable_reporting (Xtest, ytest, "hold_out_validation_set", accuracy="euclidean")
    ens.fit (Xtrain, ytrain)
    predicted = ens.predict (Xtest)
    loss = _euclidean_loss (ytest, predicted)

    print ("avg ensemble euclidean loss on validation:", loss)

    ens.write_constituent_vs_ensamble_report (Xtest, ytest, dataset_name="hold_out_validation_set")

main()
