
import os
from datetime import datetime

import numpy as np
import tqdm

from functions import *

class Ensambler:

    def __init__ (self, base_models, models_names = None, verbose=False):
        self.models = base_models
        self.verbose = verbose
        self.names = models_names
        if models_names is None:
            self.names = ["model"+str(i) for i in range (len(self.models))]
    
    def get_params ( self ):
        return dict ((name, model.get_params()) for name, model in zip(self.names, self.models))
    
    def enable_reporting ( self, X_reporting, y_reporting, dataset_name, accuracy=None, fname_prefix="" ):
        timestamp = datetime.today().isoformat().replace(':','_')
        folder_name = "ensamble_" + fname_prefix + timestamp + "_" + dataset_name
        os.makedirs ("reports/" + folder_name, exist_ok=True)
        for model_name, model in zip(self.names, self.models):
            fname = folder_name+"/"+model_name+".tsv"
            model.enable_reporting ( X_reporting, y_reporting, dataset_name, accuracy, fname )
    
    def write_constituent_vs_ensamble_report (self, X, y, accuracy="euclidean", dataset_name="n.d.", foldername=None):
        self._dataset_name = dataset_name
        self.timestamp = datetime.today().isoformat().replace(':','_')
        if foldername is None:
            foldername = "report_" + self.timestamp + "_" + dataset_name
        
        self._report_accuracy_fun_name = accuracy
        assert accuracy in accuracy_functions, "accuracy function {} not implemented".format(accuracy)
        self._report_accuracy = accuracy_functions[accuracy]

        self._report_folder = "ensamble_reports/"+foldername
        os.makedirs (self._report_folder, exist_ok=True)

        models_predictions = np.array ( [model.predict (X) for model in self.models] )
        ensamble_predictions = np.mean (models_predictions, axis=0)

        report_fname = self._report_folder + "/scores.tsv"
        with open(report_fname, "w") as report_fout:
            print ("# ensamble of {} models".format(len(self.models)), file=report_fout)
            print ("# date: {}".format(self.timestamp), file=report_fout)
            print ("# dataset: {}".format(self._dataset_name), file=report_fout)
            print ("# parameters: {}".format(self.get_params()), file=report_fout)
            print ("model name\t{}".format(self._report_accuracy_fun_name), file=report_fout)
            
            for name, predictions in zip (self.names, models_predictions):
                predictions_fname = self._report_folder + "/predictions_"+name+".tsv"
                with open (predictions_fname, "w") as predictions_fout:
                    print ("# predictions for {}".format(name), file=predictions_fout)
                    print ("\n".join ("\t".join(str(x) for x in row) for row in predictions), file=predictions_fout)
                score = self._report_accuracy (y, predictions)
                print ("{}\t{}".format(name, score), file=report_fout)
            
            predictions_fname = self._report_folder + "/predictions_ensamble.tsv"
            with open (predictions_fname, "w") as predictions_fout:
                print ("# predictions for ensamble", file=predictions_fout)
                print ("\n".join ("\t".join(str(x) for x in row) for row in ensamble_predictions), file=predictions_fout)
            score = self._report_accuracy (y, ensamble_predictions)
            print ("{}\t{}".format("ensamble", score), file=report_fout)

    def predict ( self, X ):
        models_predictions = np.array ( [model.predict (X) for model in self.models] )
        ensamble_predictions = np.mean (models_predictions, axis=0)
        return ensamble_predictions
    
    def fit ( self, X, y ):
        if self.verbose:
            iterator = tqdm.tqdm (self.models, desc="ensamble fit")
        else:
            iterator = self.models

        for model in iterator:
            model.fit (X,y)

    
