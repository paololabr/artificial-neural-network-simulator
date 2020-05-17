
import os
from datetime import datetime
import itertools
import json

import numpy as np
import tqdm

from functions import *
from utility import CreateLossPlot

class Ensembler:
    '''
        implements a model that ensemble many basic "constituent" models.
        the prediction for the ensemble are the average of the predictions of the constituent models.
    '''

    def __init__ (self, base_models, models_names = None, verbose=False):
        '''
            initializa the Enemble model given the base models and their names. If the names are not specified they default to "model0", "model1", ...

            the verbose flag shows a progress bar during the fit() operation
        '''
        self.models = base_models
        self.verbose = verbose
        self.names = models_names
        if models_names is None:
            self.names = ["model"+str(i) for i in range (len(self.models))]
    
    def get_params ( self ):
        '''
            Returns a dictionary which keys are the model names and which values are their parameters.
        '''
        return dict ((name, model.get_params()) for name, model in zip(self.names, self.models))
    
    def enable_reporting ( self, X_reporting, y_reporting, dataset_name, accuracy=None, fname_prefix="" ):
        '''
            Tells the model to create a report when the function fit() will be called.
            
            The reporting on each constituent model will be enabled using the parameters X_reporting, y_reporting, dataset_name and accuracy.
            fname_prefix will be used as a prefix for the report filenames.

        '''
        timestamp = datetime.today().isoformat().replace(':','_')
        folder_name = "ensemble_" + fname_prefix + timestamp + "_" + dataset_name
        os.makedirs ("reports/" + folder_name, exist_ok=True)
        for model_name, model in zip(self.names, self.models):
            fname = folder_name+"/"+model_name+".tsv"
            model.enable_reporting ( X_reporting, y_reporting, dataset_name, accuracy, fname )
    
    def write_constituent_vs_ensemble_report (self, X, y, accuracy="euclidean", dataset_name="n.d.", foldername=None):
        '''
            Computes the predictions for each constituent model and writes them into different files inside the same folder. 
            Also computes the value of any accuracy function for all the constituent models and the ensemble models computed on a dataset X.
            These values are written in a report file.

            :param: X dataset on which the accuracy have to be computed. It shape is (n_sample, n_features)
            :param: y true outputs for the dataset X. Its shape is (n_samples, n_outputs)
            :param: accuracy the name of the accuracy function
            :param: dataset_name an arbittrary name associated to the dataset X
            :param: foldername folder on which to put the produced files.

        '''

        self._dataset_name = dataset_name
        self.timestamp = datetime.today().isoformat().replace(':','_')
        if foldername is None:
            foldername = "report_" + self.timestamp + "_" + dataset_name
        
        self._report_accuracy_fun_name = accuracy
        assert accuracy in accuracy_functions, "accuracy function {} not implemented".format(accuracy)
        self._report_accuracy = accuracy_functions[accuracy]

        self._report_folder = "ensemble_reports/"+foldername
        os.makedirs (self._report_folder, exist_ok=True)

        models_predictions = np.array ( [model.predict (X) for model in self.models] )
        ensemble_predictions = np.mean (models_predictions, axis=0)

        report_fname = self._report_folder + "/scores.tsv"
        with open(report_fname, "w") as report_fout:
            print ("# ensemble of {} models".format(len(self.models)), file=report_fout)
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
            
            predictions_fname = self._report_folder + "/predictions_ensemble.tsv"
            with open (predictions_fname, "w") as predictions_fout:
                print ("# predictions for ensemble", file=predictions_fout)
                print ("\n".join ("\t".join(str(x) for x in row) for row in ensemble_predictions), file=predictions_fout)
            score = self._report_accuracy (y, ensemble_predictions)
            print ("{}\t{}".format("ensemble", score), file=report_fout)

    def predict ( self, X ):
        models_predictions = np.array ( [model.predict (X) for model in self.models] )
        ensemble_predictions = np.mean (models_predictions, axis=0)
        return ensemble_predictions
    
    def fit ( self, X, y ):
        if self.verbose:
            iterator = tqdm.tqdm (self.models, desc="ensemble fit")
        else:
            iterator = self.models

        for model in iterator:
            model.fit (X,y)

    def fit_and_plot_final_model_performances ( self, X, y, X_reporting, y_reporting, dataset_name, loss, fname="" ):
        '''
            Trains the consituent models epoch by epoch using the dataset X.
            
            Writes a report file containing:
             (1) the loss value for the ensemble model at each epoch (learning curve) on the dataset X;
             (1) the loss value for the ensemble model at each epoch (learning curve) on another dataset X_reporting.
            
            A png file containing the plot of the mentioned curves is also produced.

            :param: X dataset to fit the constituent models. Its shape is (n_samples, n_features)
            :param: y true values for X. Its shape is (n_samples, n_outputs)
            :param: X_reporting dataset on which the loss has to be computed. Its shape is (n_samples_reporting, n_features)
            :param: y_reporting true values for X_reporting. Its shape is (n_samples_reporting, n_outputs)
            :param: dataset_name an arbitrary name associated to X_reporting
            :param: loss name of the loss function
            :fname: name of the report filename. If not provided a name that contains the current timestamp and dataset name will be used.
            
        '''
        
        assert loss in loss_functions, "loss function {} not implemented".format(loss)
        loss_fun = loss_functions[loss]

        generators = {name: model.fit_iterator(X,y) for name, model in zip (self.names, self.models)}       
        trained_models = {}
        n_converged = 0
        
        timestamp = datetime.today().isoformat().replace(':','_')    
        if not fname:
            fname = timestamp + "_ENSEMBLE_" + dataset_name + ".tsv"

        with open("reports/" + fname, "w") as fout:
            
            print ("# ensemble of {} models".format(len(self.models)), file=fout)
            print ("# dataset:", dataset_name, file=fout)
            print ("# date:", timestamp, file=fout)
            print ("# parameters:", json.dumps (self.get_params()), file=fout)
            print ("epoch\ttrain_loss({})\tvalid_loss({})".format(loss, loss), file=fout)
            
            epoch_no = 1

            if self.verbose:
                progressbar = tqdm.tqdm (desc="epoch")

            while not n_converged == len(self.models):

                if self.verbose:
                    progressbar.set_postfix_str ("converged: {}".format(n_converged))
                    progressbar.update()

                #train all the models for one epoch
                for name in tqdm.tqdm(generators, desc="models", disable=not self.verbose):
                    try:
                        trained_models[name] = next(generators[name])
                    #if the model has converged replace its generator with one that always yields its last value (taken from the previous epoch)
                    except StopIteration:
                        generators[name] = itertools.cycle ([trained_models[name]])
                        n_converged += 1
                        # print ("epoch {}: {} converged".format(epoch_no, name))
                        # print ("Remaining generators: ", generators)
                        # input()
                
                assert len(trained_models) == len(generators) == len(self.models), "Wrong number of generators or trained models"
                
                models_predictions = np.array ( [model.predict (X) for model in self.models] )
                ensemble_predictions = np.mean (models_predictions, axis=0)
                losses_matrix = loss_fun (y, ensemble_predictions)
                train_loss = np.average (np.sum(losses_matrix, axis=1))

                models_predictions = np.array ( [model.predict (X_reporting) for model in self.models] )
                ensemble_predictions = np.mean (models_predictions, axis=0)
                losses_matrix = loss_fun (y_reporting, ensemble_predictions)
                valid_loss = np.average (np.sum(losses_matrix, axis=1))

                print (str(epoch_no) + "\t" + str(train_loss) + "\t" + str(valid_loss), file=fout)

                #DEBUG 
                # if epoch_no < 3:
                #     print ("epoch no", epoch_no)
                #     for i,pred in enumerate(models_predictions):
                #         print ("model", i, "predictions on internal test set")
                #         print (pred)
                    
                epoch_no += 1


        CreateLossPlot ("reports/" + fname)




