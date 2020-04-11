import sys

from utility import *
from neural_network import *
from functions import _euclidean_loss

def main ():

    n_best = 10
    if len(sys.argv) > 1:
        n_best = int (sys.argv[1])
    
    format = "tsv"
    if len(sys.argv) > 2:
        format = sys.argv[2]

    folder = "model_selecton_results"
    if len(sys.argv) > 3:
        folder = sys.argv[3]

    fileprefix = ""

    results = getBestRes (fileprefix, folder, n_best)

    Xtrain, ytrain, Xtest, ytest = ReadData("cup/ML-CUP19-TR.csv", 0.90)
    Xtrain, Xtest, ytrain, ytest = np.array(Xtrain), np.array(Xtest), np.array(ytrain), np.array(ytest)
    
    with open(folder + "/" + "best.tsv", 'w', buffering=1) as outt:
        for params, perf in results:
            if len(perf) == 3:
                nn = MLPRegressor ( random_state=42 )
                nn.set_params (**params)
                '''
                nn.fit (Xtrain, ytrain)
                predicted = nn.predict (Xtrain)
                loss = _euclidean_loss (ytrain, predicted)
                '''
                res=cross_val(nn,Xtrain, ytrain,_euclidean_loss,5)
                perf.insert(1, res[1])
        
            json.dump(params, outt)
            print (file=outt)
            json.dump(perf, outt)
            print (file=outt)

if __name__ == "__main__":
    main()
