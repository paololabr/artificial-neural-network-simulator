
import sys
from neural_network import *
from functions import _classification_loss
from utility import readMonk, save_table, save_grid_table
from utility import GridSearchCV
from utility import getRandomParams
import statistics

def main():

    monk_num = 3
    trials_num = 10

    monk_task = "monks-" + str(monk_num)
    Xtrain, ytrain, _, _ = readMonk("monks/" + monk_task + ".train")
    Xtest, ytest, _, _ = readMonk("monks/" +  monk_task + ".test")
    
    if len(Xtrain) == 0 or len(Xtest) == 0:
        print ("Error reading data")
        exit()

    nn = MLPClassifier( max_iter=500, n_iter_no_change=10)

    # funzionanti da Paolo (curva stabile, alpha>0 e n_iter_no_change alto)
    # params={"hidden_layer_sizes": [6], "alpha": 0.003, "activation": "tanh", "learning_rate": "constant", "learning_rate_init": 0.82,  "momentum": 0.6, "n_iter_no_change": 80}
    #params={"hidden_layer_sizes": [6], "alpha": 0.003, "activation": "tanh", "learning_rate": "constant", "learning_rate_init": 0.82,  "momentum": 0.6, "n_iter_no_change": 80}
        
    # monks1
    if (monk_num == 1):
        params={"hidden_layer_sizes": [15], "alpha": 0., "activation": "relu", "learning_rate": "constant", "learning_rate_init": 0.8,  "momentum": 0.8, "n_iter_no_change": 80, "output_activation":"logistic", "weights_init_value":0.1 }
    elif (monk_num == 2): # monks2
        params={"hidden_layer_sizes": [15], "alpha": 0., "activation": "relu", "learning_rate": "constant", "learning_rate_init": 0.8,  "momentum": 0.8, "n_iter_no_change": 80, "output_activation":"logistic", "weights_init_value":0.25 }
    elif (monk_num == 3): # monks3
        #params={"hidden_layer_sizes": [15], "alpha": 0., "activation": "relu", "learning_rate": "constant", "learning_rate_init": 0.8,  "momentum": 0.8, "n_iter_no_change": 10, "output_activation":"logistic", "weights_init_value":0.1}
        # monks3 reg
        params={"hidden_layer_sizes": [15], "alpha": 0.003, "activation": "relu", "learning_rate": "constant", "learning_rate_init": 0.8,  "momentum": 0.8, "n_iter_no_change": 10, "output_activation":"logistic", "weights_init_value":0.1}
  
    nn.set_params (**params)

    res = []
    
    for _ in range(trials_num):
        nn.enable_reporting ( Xtest, ytest, monk_task, "classification" )
        
        nn.fit (Xtrain, ytrain)
        last_row = nn._last_row

        '''
        predicted = nn.predict (Xtest)
        accuracy = _classification_loss (ytest, predicted)

        print ("avg classification accuracy on validation:", accuracy)

        predicted = nn.predict (Xtrain)
        accuracy = _classification_loss (ytrain, predicted)
        '''

        res_row = [float(x) for x in last_row.split('\t')][1:]
        res.append(res_row)
  
    train_loss = statistics.mean([x[0] for x in res])
    valid_loss = statistics.mean([x[1] for x in res])
    
    valid_acc = 100 - 100 * statistics.mean([x[2] for x in res])
    train_acc = 100 - 100 * statistics.mean([x[3] for x in res])

    fout = open (monk_task + ".txt", "w")
    print ("train_loss: {}\tvalid_loss: {}\tvalid_acc: {}\ttrain_acc: {}".format(train_loss, valid_loss, valid_acc, train_acc),file=fout)

main()
