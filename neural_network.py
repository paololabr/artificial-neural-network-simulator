
import numpy as np
from functions import activation_functions, activation_functions_derivatives, loss_functions, loss_functions_derivatives

class BaseNeuralNetwork:

    def __init__(self, hidden_layer_sizes=(100, ), hidden_activation='relu', output_activation="identity", solver='adam', alpha=0.0001, batch_size='auto',
                       learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                       random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                       early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10,
                       max_fun=15000, loss="squared"):

        self.hidden_layer_sizes = hidden_layer_sizes
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
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
        self._debug_backward_pass = False

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

        # TODO: actual learning rate rule
        self._eta = learning_rate_init

        self._weights = None

    def set_weights ( self, weights ):
        for i in range (len(weights)-1):
            assert weights[i].shape[1] == weights[i+1].shape[0]-1, "weight shapes must be compatible. Got {}".format(list(map (lambda x: x.shape, weights)))
        self._weights = weights

    def _generate_random_weights ( self, n_features, n_outputs ):
        self._weights = []
        for n,m in zip ([n_features]+list(self.hidden_layer_sizes), list(self.hidden_layer_sizes)+[n_outputs]):
            W = 0.7 * np.random.randn (n+1,m)
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
        # TODO: use minibatch and shuffle according to parameters
        layers_nets, layer_outputs = self._forward_pass (X)
        delta_weights = self._backpropagation ( layers_nets, layer_outputs, y )
        for W, dW in zip (self._weights, delta_weights):
            # TODO: use the right learning rate depending on the epochs
            W -= self._eta * dW

    def predict ( self, X ):
        assert self._weights is not None, "call fit() or set_weights() before predict()"
        X = np.array (X)
        _ , layer_outputs = self._forward_pass (X)
        return layer_outputs[-1]
    
    def fit ( self, X, y ):
        X = np.array (X)
        y = np.array (y)
        if not self._weights or not self.warm_start:
            self._generate_random_weights (X.shape[1], y.shape[1])

        # TODO: other stopping criterions
        epoch_no = 0
        while epoch_no < self.max_iter:
            self._do_epoch ( X, y )
            predicted = self.predict (X)
            loss = self._loss (predicted, y)
            # print ("Loss for epoch {}: {}".format(epoch_no, sum(loss)))
            epoch_no += 1


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
    print ("\n\n\n\n")

    n = BaseNeuralNetwork ( hidden_layer_sizes=(2,), hidden_activation="logistic", output_activation="logistic", max_iter=1, 
                            warm_start=True, learning_rate_init=0.5 )
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
    y = np.array ([[0.01, 0.99]])
    # n._debug_forward_pass = True
    # n._debug_backward_pass = True
    predicted = n.predict (X)
    losses = loss_functions["squared"] (predicted, y)
    print ("TEST backpropagation (matt mazur example: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)")
    for x, p, loss in zip (X,predicted, losses):
        print ("MATTMAZ({}) = {} - loss: {}".format(x,p, sum(loss)))
    n.fit (X, y)
    print ("weights after running one epoch:")
    print (n._weights)
    print ("\n\n\n\n")

    n = BaseNeuralNetwork (hidden_layer_sizes=(50,), learning_rate_init=0.01)
    X = [ [0, 0],
          [0, 1],
          [1, 0],
          [1, 1]
    ]
    y = [ [0.99],
          [0.01],
          [0.01],
          [0.99]
    ]
    n.fit (X, y)
    predicted = n.predict (X)
    losses = loss_functions["squared"] (predicted, y)
    print ("TEST network that learns how to compute XnOR")
    for x, p, loss in zip (X,predicted, losses):
        print ("XnOR({}) = {} - loss: {}".format(x,p, sum(loss)))
    print ("\n\n\n\n")
    
    n = BaseNeuralNetwork (hidden_layer_sizes=(50,), learning_rate_init=0.01)
    X = [ 
          [-0.55609785, -0.44237751, -1.51930792,  0.31342967],
          [ 1.86589251, -0.64794613, -1.40532609,  0.19970042],
          [ 2.07525975,  1.14612304, -1.21620428, -0.2127494 ],
          [-0.9680726 ,  1.81546847, -0.71370392, -0.37450352]
        ]
    y = [
          [-2.20435361, -0.88475503 ],
          [ 0.0123207,  -1.29589226 ],
          [ 1.79242911,  2.29224607 ],
          [-0.24081157,  3.63093693 ]
        ]
    n.fit (X, y)
    predicted = n.predict (X)
    losses = loss_functions["squared"] (predicted, y)
    print ("TEST network that learns to compute sum of its inputs and double of the second input")
    for x, p, loss in zip (X,predicted, losses):
        print ("sum and double({}) = {} - loss: {}".format(x,p, sum(loss)))
