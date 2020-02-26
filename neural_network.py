
import numpy as np
from functions import activation_functions

class BaseNeuralNetwork:

    #TODO: hidden activation vs output activation
    def __init__(self, hidden_layer_sizes=(100, ), hidden_activation='relu', output_activation="identity", solver='adam', alpha=0.0001, batch_size='auto',
                       learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                       random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                       early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10,
                       max_fun=15000):

        self.hidden_layer_sizes = hidden_layer_sizes
        self.solver = solver
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
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun

        self._debug_forward_pass = False

        if hidden_activation not in activation_functions:
            raise ValueError ("hidden activation function {} not implemented".format(hidden_activation))
        self._hidden_activation = activation_functions[hidden_activation]

        if output_activation not in activation_functions:
            raise ValueError ("output activation function {} not implemented".format(output_activation))
        self._output_activation = activation_functions[output_activation]

        self._weights = None

    def set_weights ( self, weights ):
        for i in range (len(weights)-1):
            assert weights[i].shape[1] == weights[i+1].shape[0]-1, "weight shapes must be compatible. Got {}".format(list(map (lambda x: x.shape, weights)))
        self._weights = weights

    def _forward_pass ( self, X ):
        '''
            feeds the network with a minibatch of samples X (n_samples, n_features)
            returns layer_nets, outputs such that:
                layer_nets[0] is the input X (n_samples, n_features)
                layer_nets[i] for i=1...n_layers are the nets of the hidden layer i (n_samples, dim_layer_i)
                outputs are the predicted values (n_samples, n_classes)
        '''
        assert X.shape[1] == self._weights[0].shape[0]-1, "wrong number of features {} for first layer weights shape {}".format(X.shape[1], self.weights[0].shape[0]-1)
        prev_level_output = X
        layer_nets = [X]
        
        #hidden layers
        for i in range (len(self._weights)-1):
            biases = np.ones( (prev_level_output.shape[0], 1) )
            inp_and_biases = np.hstack ( (prev_level_output, biases) )
            net = np.matmul (inp_and_biases, self._weights[i])
            layer_nets.append (net)
            prev_level_output = self._hidden_activation (net)
            if self._debug_forward_pass:
                print ("[DEBUG] layer {}\ninput + bias:\n{}\nweights\n{}\nnet\n{}\noutput\n{}".format(i, inp_and_biases, self._weights[i], net, prev_level_output))
        
        # output layer
        biases = np.ones( (prev_level_output.shape[0], 1) )
        inp_and_biases = np.hstack ( (prev_level_output, biases) )
        net = np.matmul (inp_and_biases, self._weights[-1])
        layer_nets.append (net)
        output =  self._output_activation (net)
        
        if self._debug_forward_pass:
            print ("[DEBUG] output layer\ninput + bias:\n{}\nweights\n{}\nnet\n{}\noutput\n{}".format(inp_and_biases, self._weights[-1], net, output))
        
        return layer_nets, output

    def predict ( self, X ):
        assert self._weights is not None, "call fit() or set_weights() before predict()"
        _ , output = self._forward_pass (X)
        return output



#unit tests
if __name__ == "__main__":
    # a network that XORs its inputs
    n = BaseNeuralNetwork ( hidden_layer_sizes=(2,), hidden_activation="threshold" )
    weights = [
                np.array ([[ 1,    1  ],
                           [ 1,    1  ],
                           [-1.5, -0.5]]),
                np.array ([[-1  ],
                           [ 1  ],
                           [-0.5]])
    ]
    n.set_weights (weights)
    X = np.array ([[0,0],
                   [0,1],
                   [1,0],
                   [1,1]])
    predicted = n.predict (X)
    print ("TEST forward pass (XOR)")
    for x, y in zip (X,predicted):
        print ("XOR({}) = {}".format(x,y))

    n = BaseNeuralNetwork ( hidden_layer_sizes=(2,), hidden_activation="logistic", output_activation="logistic" )
    weights = [
                np.array ([[ .15, .25 ],
                           [ .2 , .3  ],
                           [ .35, .35 ]]),
                np.array ([[ .4 , .5  ],
                           [ .45, .55 ],
                           [ .6 , .6  ]])
    ]
    n.set_weights (weights)
    X = np.array ([[0.05, 0.1]])
    # n._debug_forward_pass = True
    predicted = n.predict (X)
    print ("TEST backpropagation (matt mazur example: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)")
    for x, y in zip (X,predicted):
        print ("MATTMAZ({}) = {}".format(x,y))
    
