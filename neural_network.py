
import numpy as np
import random
from datetime import datetime
import json
import os
import copy
from utility import CreateLossPlot, CreateAccuracyPlot

from functions import activation_functions, activation_functions_derivatives, loss_functions, loss_functions_derivatives, accuracy_functions, weights_init_functions
from sklearn.model_selection import train_test_split

np.seterr (all="raise", under="ignore")

class BaseNeuralNetwork:

    def __init__(self, hidden_layer_sizes=(100, ), hidden_activation='relu', output_activation="identity", solver='sgd', alpha=0.0001, batch_size='auto',
                       learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                       random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                       early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10,
                       max_fun=15000, loss="squared", weights_init_fun = "random_normal", weights_init_value=0.7 ):

        # configurable parameters
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.activation = hidden_activation

        if weights_init_fun not in weights_init_functions:
            raise ValueError ("weights init. function {} not implemented".format(weights_init_functions))

        self.weights_init_fun = weights_init_fun
        self.weights_init_value = weights_init_value
        self.weights_init_function = weights_init_functions[weights_init_fun]

        # fixed parameters
        self.linear_decay_iterations = 100
        self.linear_decay_eta_zero = learning_rate_init / 100

        # debug and reporting flags
        self._debug_forward_pass = False
        self._debug_backward_pass = False
        self._debug_epochs = False
        self._debug_early_stopping = False
        self._do_reporting = False
        self._debug_report = False

        # external readable properties
        self.out_activation_ = output_activation

        self.b_size = 0

        if solver != "sgd":
            raise ValueError ("Only stochastic gradient descent solver is implemented")

        if hidden_activation not in activation_functions or hidden_activation not in activation_functions_derivatives:
            raise ValueError ("hidden activation function {} not implemented".format(hidden_activation))
        self._hidden_activation = activation_functions[hidden_activation]
        self._hidden_activation_derivative = activation_functions_derivatives[hidden_activation]

        if output_activation not in activation_functions or output_activation not in activation_functions_derivatives:
            raise ValueError ("output activation function {} not implemented".format(output_activation))
        self._output_activation = activation_functions[output_activation]
        self._output_activation_derivative = activation_functions_derivatives[output_activation]

        if loss not in loss_functions or loss not in loss_functions_derivatives:
            raise ValueError ("loss function {} not implemented".format(loss_functions))
        self._loss = loss_functions[loss]
        self._loss_derivative = loss_functions_derivatives[loss]
        self._loss_fun_name = loss

        if random_state is not None:
            np.random.seed ( random_state )

        self._weights = None
        self.delta_olds = None
    
    def get_params (self, deep=True):
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "alpha": self.alpha,
            "batch_size": self.batch_size,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "learning_rate_init": self.learning_rate_init,
            "power_t": self.power_t,
            "max_iter": self.max_iter,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "tol": self.tol,
            "verbose": self.verbose,
            "warm_start": self.warm_start,
            "momentum": self.momentum,
            "nesterovs_momentum": self.nesterovs_momentum,
            "early_stopping": self.early_stopping,
            "validation_fraction": self.validation_fraction,
            "n_iter_no_change": self.n_iter_no_change,
            "weights_init_fun":  self.weights_init_fun,
            "weights_init_value": self.weights_init_value
        }
    
    def set_params (self, **parameters_dict):
        for param in ["hidden_layer_sizes", "alpha", "n_iter_no_change", "validation_fraction", "early_stopping", "nesterovs_momentum", "momentum", "warm_start", "verbose", "tol", "random_state", "shuffle", "max_iter", "power_t", "learning_rate_init", "learning_rate", "activation", "batch_size", "weights_init_fun", "weights_init_value"  ]:
            if param in parameters_dict:
                setattr (self, param, parameters_dict[param])
 
    def set_weights ( self, weights ):
        for i in range (len(weights)-1):
            assert weights[i].shape[1] == weights[i+1].shape[0]-1, "weight shapes must be compatible. Got {}".format(list(map (lambda x: x.shape, weights)))
        self._weights = weights

    def _check_fit_datasets (self, X, y):
        X = np.array (X)
        y = np.array (y)
        # if y.shape == (n_samples) convert it to a column vector (n_samples, 1)
        if y.ndim == 1:
            y = y[:, np.newaxis]

        assert len(X) == len(y), "size of X and y must be the same"
        return X, y


    def enable_reporting ( self, X_reporting, y_reporting, dataset_name, accuracy=None, fname=None ):
        self._do_reporting = True
        self.X_reporting, self.y_reporting = self._check_fit_datasets (X_reporting, y_reporting)
        self.timestamp = datetime.today().isoformat().replace(':','_')
        if fname is None:
            fname = self.timestamp + "_" + dataset_name + ".tsv"
        
        self._report_accuracy = None
        self._last_row = ""
        if accuracy is not None:
            self._report_accuracy_fun_name = accuracy
            assert accuracy in accuracy_functions, "accuracy function {} not implemented".format(accuracy)
            self._report_accuracy = accuracy_functions[accuracy]

        os.makedirs ("reports", exist_ok=True)

        self._report_dataset_name = dataset_name
        self._report_fname = "reports/"+fname

    def _write_report_header ( self, fout ):
        print ("#", type(self).__name__, file=fout)
        print ("# dataset:", self._report_dataset_name, file=fout)
        print ("# date:", self.timestamp, file=fout)
        print ("# parameters:", json.dumps (self.get_params()), file=fout)
        header_row = "epoch\ttrain_loss({})\tvalid_loss({})".format(self._loss_fun_name, self._loss_fun_name)
        if self._report_accuracy:
            header_row += "\tvalid_accuracy({})\ttrain_accuracy({})".format(self._report_accuracy_fun_name, self._report_accuracy_fun_name )
        print (header_row, file=fout)
        
    def _write_report_epoch ( self, fout, epoch_no, train_loss, train_accuracy ):
        predicted = self._predict_internal (self.X_reporting)
        losses_matrix = self._loss (self.y_reporting, predicted)
        valid_loss = np.average (np.sum(losses_matrix, axis=1))
 
        self._last_row  = str(epoch_no) + "\t" + str(train_loss) + "\t" + str(valid_loss)
        
        if self._report_accuracy:
            predicted = self.predict (self.X_reporting)
            valid_accuracy = self._report_accuracy (self.y_reporting, predicted)
            self._last_row += "\t" + str (valid_accuracy) + "\t" + str (train_accuracy)
        
        print (self._last_row, file=fout)

        if self._debug_report:
            print ("[DEBUG REPORT] **** epoch {} ******".format(epoch_no))
            print ("y_reporting")
            print (self.y_reporting)
            print ("predictions for y_reporting")
            print (predicted)
            print ("losses matrix")
            print (losses_matrix)
            print ("validation loss ({}): {}".format (self._loss_fun_name, valid_loss))
            print ("validation accuracy({}): {}".format (self._report_accuracy_fun_name, valid_accuracy))
            
            # input ("press enter to continue...")


    def _generate_random_weights ( self, n_features, n_outputs ):
        self._weights = []
        for n,m in zip ([n_features]+list(self.hidden_layer_sizes), list(self.hidden_layer_sizes)+[n_outputs]):
            #W = 0.7 * np.random.randn (n+1,m)
            W = self.weights_init_function(self.weights_init_value, (n+1,m))
            self._weights.append (W)

    def _forward_pass ( self, X ):
        '''
            feeds the network with a minibatch of samples X (n_samples, n_features)
            returns layer_nets, layer_outputs such that:
                layer_nets[0] is the input X (n_samples, n_features)
                layer_nets[i] for i=1...n_layers are the nets of the hidden layer i (n_samples, dim_layer_i)
                layer_outputs[0] is the input X (n_samples, n_features)
                layer_outputs[i] for i=1...n_layers are the outputs of the hidden layer i (n_samples, dim_layer_i)

                in particular, layer_outputs[-1] is the output predicted by the output units 
        '''
        assert X.shape[1] == self._weights[0].shape[0]-1, "wrong number of features {} for first layer weights shape {}".format(X.shape[1], self._weights[0].shape[0]-1)
        layer_outputs = [X]
        layer_nets = [X]
        
        #hidden layers
        for i in range (len(self._weights)-1):
            inp = layer_outputs[i]
           
            biases = np.ones( (inp.shape[0], 1) )
            inp_and_biases = np.hstack ( (inp, biases) )
            
            net = np.matmul (inp_and_biases, self._weights[i])
            
            layer_nets.append (net)
            layer_outputs.append ( self._hidden_activation (net) )
            if self._debug_forward_pass:
                print ("[DEBUG] layer {}\ninput + bias:\n{}\nweights\n{}\nnet\n{}\noutput\n{}".format(i, inp_and_biases, self._weights[i], net, layer_outputs[-1]))
        
        # output layer
        inp = layer_outputs[-1]
        biases = np.ones( (inp.shape[0], 1) )
        inp_and_biases = np.hstack ( (inp, biases) )
        net = np.matmul (inp_and_biases, self._weights[-1])
        layer_nets.append (net)
        layer_outputs.append (self._output_activation (net))
        
        if self._debug_forward_pass:
            print ("[DEBUG] output layer\ninput + bias:\n{}\nweights\n{}\nnet\n{}\noutput\n{}".format(inp_and_biases, self._weights[-1], net, layer_outputs[-1]))
        
        return layer_nets, layer_outputs

    def _backpropagation ( self, layers_nets, layers_outputs, real_outputs ):

        assert len(layers_outputs) == len(layers_nets), "Backpropagation: number of layers outputs and nets must be the same."
        assert len(layers_outputs) == len(self._weights) + 1, "Backpropagation: number of layers outputs must be number of weights matrices + 1"

        n_samples = len(layers_outputs[0])
        delta_weights = []

        #output layers        
        dE = self._loss_derivative (real_outputs, layers_outputs[-1])
        df = self._output_activation_derivative ( layers_nets[-1] )
        deltas = dE * df

        prev_layer_outputs = layers_outputs[-2]
        biases = np.ones( (prev_layer_outputs.shape[0], 1) )
        out_and_biases = np.hstack ( (prev_layer_outputs, biases) )
        dW = sum ( a[:, np.newaxis]*b for a,b in zip (out_and_biases, deltas) ) / n_samples

        delta_weights.insert (0, dW)

        if self._debug_backward_pass:
            print ("[DEBUG] backpropagation output layer\nerror derivative:\n{}\nactivation derivative\n{}".format(dE, df))
            print ("[DEBUG] deltas for this layer (=dE*df)\n{}".format(deltas))
            print ("[DEBUG] dW\n{}".format(dW))
            print ("\n")
        
        # hidden layers
        for i in range ( len(layers_outputs)-2, 0, -1 ):
            weights = self._weights[i]
            dE = np.array ( list (map(lambda d: np.matmul(weights,d[:,np.newaxis]).flatten(), deltas )) )
            # remove dE/do_b where o_b is the bias output
            dE = dE[:, :-1]
            df = self._hidden_activation_derivative ( layers_nets[i] )
            deltas = dE * df
            prev_layer_outputs = layers_outputs[i-1]
            biases = np.ones( (prev_layer_outputs.shape[0], 1) )
            out_and_biases = np.hstack ((prev_layer_outputs, biases))
            dW = sum ( a[:, np.newaxis]*b for a,b in zip (out_and_biases, deltas) ) / n_samples
            delta_weights.insert (0, dW)

            if self._debug_backward_pass:
                print ("[DEBUG] backpropagation layer {}\nerror derivative:\n{}\nactivation derivative\n{}".format(i, dE, df))
                print ("[DEBUG] deltas for this layer (=dE*df)\n{}".format(deltas))
                print ("[DEBUG] dW\n{}".format(dW))
                print ("\n")


        assert len(delta_weights) == len (self._weights), "Backpropagation: number of delta_weights and weights are not the same"     
        return delta_weights

    def _do_epoch ( self, X, y ):

        if (self.shuffle):
            indexes = list (range(len(X)))
            random.shuffle (indexes)
            X = X[indexes]
            y = y[indexes]

        if self.delta_olds is None:
            self.delta_olds = [np.zeros_like(W) for W in self._weights]

        n_iterations = len(X) // self.b_size + (0 if len(X) % self.b_size == 0 else 1)

        for b in range(n_iterations):

            start = self.b_size * b
            stop = self.b_size * (b + 1)

            layers_nets, layer_outputs = self._forward_pass (X[start:stop])

            delta_weights = self._backpropagation ( layers_nets, layer_outputs, y[start:stop] )
            for W, dW, m in zip (self._weights, delta_weights, self.delta_olds):
                
                m *= self.momentum
                m += (1 - self.momentum) * dW

                W -= (self._eta * m) + 2 * (self.alpha * (self.b_size/len(X)) * W)
     
    def _predict_internal ( self, X ):
    # for internal usage only, subclasses can reimplement predict() instead
        assert self._weights is not None, "call fit() or set_weights() before predict()"
        X = np.array (X)
        _ , layer_outputs = self._forward_pass (X)
        return layer_outputs[-1]

    def predict ( self, X ):
        return self._predict_internal(X)

    def fit ( self, X, y ):

        X, y = self._check_fit_datasets (X,y)

        if self.weights_init_fun not in weights_init_functions:
            raise ValueError ("weights init. function {} not implemented".format(self.weights_init_fun))

        self.weights_init_function = weights_init_functions[self.weights_init_fun]

        if not self._weights or not self.warm_start:
            self._generate_random_weights (X.shape[1], y.shape[1])

        self._eta = self.learning_rate_init

        X_validation = None
        y_validation = None
        self.delta_olds = None

        if self.activation not in activation_functions or self.activation not in activation_functions_derivatives:
            raise ValueError ("hidden activation function {} not implemented".format(self.activation))
        self._hidden_activation = activation_functions[self.activation]
        self._hidden_activation_derivative = activation_functions_derivatives[self.activation]

        if self.early_stopping:
            if self._debug_early_stopping:
                print ("[DEBUG] early stopping: (original) X.shape {} y.shape {} - validation fraction {}".format(X.shape, y.shape, self.validation_fraction))  
            
            X, X_validation, y, y_validation = train_test_split ( X, y, test_size=self.validation_fraction, shuffle=self.shuffle )
            
            if self._debug_early_stopping:
                print ("[DEBUG] early stopping (after hold out) X.shape {} y.shape {}".format(X.shape, y.shape))
                print ("[DEBUG] early stopping X_validation.shape {} y_validation.shape {}".format(X_validation.shape, y_validation.shape))

        if self._do_reporting:
            report_fout = open (self._report_fname, "w")
            self._write_report_header ( report_fout )

        epoch_no = 1

        if (self.batch_size=='auto'):
            self.b_size=min(200, len(X))
        else:
            self.b_size=max(1, min(self.batch_size, len(X)))

        if self._debug_epochs:
            n_iterations = len(X) // self.b_size + (0 if len(X) % self.b_size == 0 else 1)
            print ("[DEBUG] batch size:", self.b_size)
            print ("[DEBUG] n_iterations per epoch:", n_iterations)
        
        best_loss = np.inf
        best_weights = None
        # number of epochs since last loss improvement
        loss_not_decreasing_since_epochs = 0

        while epoch_no <= self.max_iter and loss_not_decreasing_since_epochs < self.n_iter_no_change:

            if self.learning_rate == "invscaling":
                self._eta = self.learning_rate_init / pow (epoch_no, self.power_t )

            if self.learning_rate == "linear":
                if epoch_no <= self.linear_decay_iterations:
                    alpha_decay = epoch_no / self.linear_decay_iterations
                    self._eta = (1 - alpha_decay) * self.learning_rate_init + alpha_decay * self.linear_decay_eta_zero
                else:
                    self._eta = self.linear_decay_eta_zero

            self._do_epoch ( X, y )
            
            if self.early_stopping:
                predicted = self._predict_internal (X_validation)
                losses_matrix = self._loss (y_validation, predicted)
            else:
                predicted = self._predict_internal (X)
                losses_matrix = self._loss (y, predicted)
                
            avg_loss = np.average (np.sum(losses_matrix, axis=1))

            if self._debug_epochs:
                print ("average loss for epoch {}: {}".format(epoch_no, avg_loss))
            
            if avg_loss < best_loss - self.tol:
                loss_not_decreasing_since_epochs = 0
                best_loss = avg_loss
                best_weights = copy.deepcopy (self._weights)
            else:
                loss_not_decreasing_since_epochs += 1
                # with "adaptive" learning rate if the loss does not improve for two consecutive epochs: divide learning rate by 2
                if self.learning_rate == "adaptive" and loss_not_decreasing_since_epochs % 2 == 0:
                    self._eta = self._eta/2
                    if self._debug_epochs:
                        print ("decreasing learning rate")

            if self._do_reporting:
                train_accuracy = None
                if self._report_accuracy:
                    if self.early_stopping:
                        predicted_for_report = self.predict(X_validation)
                    else:
                        predicted_for_report = self.predict(X)
                    train_accuracy = self._report_accuracy (y, predicted_for_report)
                self._write_report_epoch ( report_fout, epoch_no, avg_loss, train_accuracy )

            epoch_no += 1
        
        if self.early_stopping:
            self.set_weights (best_weights)

        # set external-readable properties after fitting
        self.n_iter_ = epoch_no
        self.loss_ = best_loss
        self.n_layers_ = len(self.hidden_layer_sizes)
        self.n_outputs_ = y.shape[1]
        self.hidden_activation_ = self.activation

        if self._do_reporting:
            report_fout.close ()
            CreateLossPlot(self._report_fname)
            if self._report_accuracy:
                CreateAccuracyPlot(self._report_fname)

    def fit_iterator ( self, X, y ):
        '''
        iterator version of fit(X,y): yields the trained model (self) at each epoch.
        does not honor debug flags nor writes reports.

        example usage:
            
        .. code-block:: python

        ...

            model = BaseNeuralNetwork ()
            X,y = read\\_datasets ("train_set")
            X_valid, y_valid = read_dataset ("validation_set")
            for epoch_no, trained_model in enumrate(model.fit_iterator (X,y)):
                p_valid = trained_model.predict (X_valid)
                validation_loss = compute_loss (y_valid, p_valid)
                print ("epich number", epoch_no)
                print ("Training loss:", trained_model.loss_)
                print ("Validation loss:", validation_loss)
        ...

        '''

        X, y = self._check_fit_datasets (X,y)

        if self.weights_init_fun not in weights_init_functions:
            raise ValueError ("weights init. function {} not implemented".format(self.weights_init_fun))

        self.weights_init_function = weights_init_functions[self.weights_init_fun]

        if not self._weights or not self.warm_start:
            self._generate_random_weights (X.shape[1], y.shape[1])

        self._eta = self.learning_rate_init

        X_validation = None
        y_validation = None
        self.delta_olds = None

        if self.activation not in activation_functions or self.activation not in activation_functions_derivatives:
            raise ValueError ("hidden activation function {} not implemented".format(self.activation))
        self._hidden_activation = activation_functions[self.activation]
        self._hidden_activation_derivative = activation_functions_derivatives[self.activation]

        if self.early_stopping:
            X, X_validation, y, y_validation = train_test_split ( X, y, test_size=self.validation_fraction, shuffle=self.shuffle )
            
        epoch_no = 1

        if (self.batch_size=='auto'):
            self.b_size=min(200, len(X))
        else:
            self.b_size=max(1, min(self.batch_size, len(X)))
        
        best_loss = np.inf
        best_weights = None
        # number of epochs since last loss improvement
        loss_not_decreasing_since_epochs = 0

        while epoch_no <= self.max_iter and loss_not_decreasing_since_epochs < self.n_iter_no_change:

            if self.learning_rate == "invscaling":
                self._eta = self.learning_rate_init / pow (epoch_no, self.power_t )

            if self.learning_rate == "linear":
                if epoch_no <= self.linear_decay_iterations:
                    alpha_decay = epoch_no / self.linear_decay_iterations
                    self._eta = (1 - alpha_decay) * self.learning_rate_init + alpha_decay * self.linear_decay_eta_zero
                else:
                    self._eta = self.linear_decay_eta_zero

            self._do_epoch ( X, y )
            
            if self.early_stopping:
                predicted = self._predict_internal (X_validation)
                losses_matrix = self._loss (y_validation, predicted)
            else:
                predicted = self._predict_internal (X)
                losses_matrix = self._loss (y, predicted)
                
            avg_loss = np.average (np.sum(losses_matrix, axis=1))

            
            if avg_loss < best_loss - self.tol:
                loss_not_decreasing_since_epochs = 0
                best_loss = avg_loss
                best_weights = copy.deepcopy (self._weights)
            else:
                loss_not_decreasing_since_epochs += 1
                # with "adaptive" learning rate if the loss does not improve for two consecutive epochs: divide learning rate by 2
                if self.learning_rate == "adaptive" and loss_not_decreasing_since_epochs % 2 == 0:
                    self._eta = self._eta/2

            # set external-readable properties after fitting
            self.n_iter_ = epoch_no
            self.loss_ = avg_loss
            self.n_layers_ = len(self.hidden_layer_sizes)
            self.n_outputs_ = y.shape[1]
            self.hidden_activation_ = self.activation

            yield self

            epoch_no += 1
        
        self._loss = best_loss

        if self.early_stopping:
            self.set_weights (best_weights)

        
class MLPRegressor (BaseNeuralNetwork):
    def __init__ ( self, hidden_layer_sizes=(100, ), activation='relu', solver='sgd', alpha=0.0001, batch_size='auto', 
                   learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, 
                   tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
                   validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000,weights_init_fun="random_normal", weights_init_value=0.7 ):
        
        super().__init__ (hidden_layer_sizes=hidden_layer_sizes, hidden_activation=activation, output_activation="identity", 
                       solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init,
                       power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose, 
                       warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping, 
                       validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change,
                       max_fun=max_fun, loss="squared", weights_init_fun=weights_init_fun, weights_init_value=weights_init_value)

class MLPClassifier (BaseNeuralNetwork):
    def __init__ ( self, hidden_layer_sizes=(100, ), activation='relu', output_activation="zero_one_tanh", solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='constant',
                   learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False,
                   warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                   beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000,weights_init_fun="random_uniform", weights_init_value=0.25 ):
        
        super().__init__ (hidden_layer_sizes=hidden_layer_sizes, hidden_activation=activation, output_activation="zero_one_tanh", 
                       solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init,
                       power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose, 
                       warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping, 
                       validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change,
                       max_fun=max_fun, loss="log_loss",weights_init_fun=weights_init_fun, weights_init_value=weights_init_value)
    
    def fit ( self, X, y ):
        y = np.array (y)
        # if y.shape == (n_samples) convert it to a column vector (n_samples, 1)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        assert y.shape[1] == 1, "Multilabel output is not supported for classification"
        for label in y[:, 0]:
            assert label == 0 or label == 1, "labels for classification must be either 0 or 1"
        self.classes_ = [0,1]
        super().fit(X,y)

    def predict ( self, X ):
        y = self._predict_internal (X)
        ones = y >= 0.5
        zeros = y < 0.5
        y[ones] = 1
        y[zeros] = 0
        return y

    def predict_proba ( self, X ):
        return self._predict_internal (X)
    
    def predict_log_proba ( self, X ):
        return np.log( self.predict_proba(X) )
