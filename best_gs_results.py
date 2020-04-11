'''
    usage:
        python best_gs_results.py [ N [format] ]
    retrieves best N gridSearch results in the folder "model_selection_results" and output them using the specified format.
    N defaults to 10
    format defaults to "tsv". It can be either "tsv" (print tab-separated-value parameters) or "dict" (print python dict literal)
'''

import sys

from utility import *

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

    print ("TOP {} RESULTS IN FOLDER {}/{}* :".format(n_best, folder, fileprefix))

    if format == "tsv":

        print ("\t".join (["rank", "avg validation loss", "avg train loss", "std_loss", "n_folds"] + [key for key in sorted (results[0][0])]))
        for i, (params, perf) in enumerate (results, 1):
            if (len(perf) == 3):
                perf.insert(1, '-')
            print ("\t".join ([str(i)] + [str(x) for x in perf] + [str(params[key]) for key in sorted (params)]))
    else:
        for i, (params, perf) in enumerate (results, 1):            
            print (params)

if __name__ == "__main__":
    main()
