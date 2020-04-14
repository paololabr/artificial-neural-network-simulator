import sys
import numpy as np
from neural_network import *
from utility import ReadData, ReadBlindData
from ensembler import Ensembler
from functions import _euclidean_loss
from datetime import date
from unit_tests import DummyModel

_OUT_FNAME = "internal_and_blind_test_results/blind_test_predictions.csv"
_HEADER = "# Lucio Messina	Paolo Labruna\n# ehNonLoSo\n# ML-CUP19\n# " + date.today().strftime("%d/%m/%Y")

def main():

    Xtrain, ytrain, _, _ = ReadData("cup/ML-CUP19-TR.csv", 1)
    ids, Xtest = ReadBlindData("cup/ML-CUP19-TS.csv")
    Xtrain, ytrain, ids, Xtest = np.array(Xtrain), np.array(ytrain), np.array(ids), np.array(Xtest)

    if (len(Xtrain) == 0 or len(Xtest) == 0):
        print ("Error reading data")
        exit()

    constituent_params = [
        {'activation': 'tanh', 'alpha': 0.0028181084594129657, 'batch_size': 1, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'linear', 'learning_rate_init': 0.005887497817889539, 'momentum': 0.6415040512902583, 'weights_init_fun': 'random_normal', 'weights_init_value': 0.3904774144322434},
        {'activation': 'logistic', 'alpha': 0.0005951915361345095, 'batch_size': 1, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'adaptive', 'learning_rate_init': 0.054980020763198, 'momentum': 0.8050419419933234},
        {'activation': 'tanh', 'alpha': 0.0014347108668793685, 'batch_size': 5, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'linear', 'learning_rate_init': 0.03018084193934551, 'momentum': 0.7819029994042981, 'weights_init_fun': 'random_uniform', 'weights_init_value': 0.39013236197037476},
        {'activation': 'tanh', 'alpha': 0.0007612380141853892, 'batch_size': 5, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'linear', 'learning_rate_init': 0.038179078019998106, 'momentum': 0.06888777870324973, 'weights_init_fun': 'random_normal', 'weights_init_value': 0.3330417022697034},
        {'activation': 'logistic', 'alpha': 0.0002188148237404819, 'batch_size': 1, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'constant', 'learning_rate_init': 0.016437178740396977, 'momentum': 0.483140617180181},
        {'activation': 'logistic', 'alpha': 0.0008463654654544185, 'batch_size': 1, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'adaptive', 'learning_rate_init': 0.07968904898581924, 'momentum': 0.2954139368874502, 'weights_init_fun': 'random_uniform', 'weights_init_value': 0.6682284952429505},
        {'activation': 'logistic', 'alpha': 0.0009044205421256557, 'batch_size': 1, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'adaptive', 'learning_rate_init': 0.06205633095090332, 'momentum': 0.01510303444629194},
        {'activation': 'logistic', 'alpha': 0.0017164735753597117, 'batch_size': 5, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'linear', 'learning_rate_init': 0.051375660832639905, 'momentum': 0.5586182890198031},
        {'activation': 'logistic', 'alpha': 0.001542764264801877, 'batch_size': 5, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'constant', 'learning_rate_init': 0.020231597719749344, 'momentum': 0.5704950000170622},
        {'activation': 'logistic', 'alpha': 0.0016753169392290158, 'batch_size': 5, 'early_stopping': False, 'hidden_layer_sizes': [50, 50], 'learning_rate': 'constant', 'learning_rate_init': 0.06569622348675933, 'momentum': 0.5544087332459164},
    ]    

    models = []
    num_conf = len(constituent_params)
    if (len(sys.argv) > 1) and  (0 < int(sys.argv[1]) < len(constituent_params)):
        num_conf = int (sys.argv[1])
    
    for index, parameters_dict in enumerate(constituent_params):
        nn = MLPRegressor ( random_state=42 )
        nn.set_params (**parameters_dict)
        models.append (nn)
        if (index == (num_conf-1)):
            break

    
    print ("Xtrain.shape", Xtrain.shape)
    print ("ytrain.shape", ytrain.shape)
    print ("ids.shape", ids.shape)
    print ("Xtest.shape", Xtest.shape)

    # ens = DummyModel () 

    ens = Ensembler (models, verbose=True)
    ens.enable_reporting (Xtrain, ytrain, "whole_dataset", accuracy="euclidean")

    ens.fit (Xtrain, ytrain)
    predicted = ens.predict (Xtest)
    
    with open (_OUT_FNAME, "w") as fout:
        print (_HEADER, file=fout)
        print ("\n".join (str(sample_id)+","+str(y[0])+","+str(y[1]) for sample_id, y in zip (ids, predicted)), file=fout)

    ens.write_constituent_vs_ensemble_report (Xtrain, ytrain, dataset_name="whole_dataset")

main()
