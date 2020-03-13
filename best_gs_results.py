
import sys

from utility import *

def main ():

    n_best = 10
    if len(sys.argv) > 1:
        n_best = int (sys.argv[1])

    folder = "model_selecton_results"
    fileprefix = ""

    results = getBestRes (fileprefix, folder, n_best)

    print ("TOP {} RESULTS IN FOLDER {}/{}* :".format(n_best, folder, fileprefix))

    for i, (params, perf) in enumerate (results, 1):
        print ("#{}".format(i))
        print ("params: {}".format(params))
        print ("[avg_loss, std_loss, n_folds]: {}".format(perf))


if __name__ == "__main__":
    main()
