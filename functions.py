
import numpy as np
import math

############################
#   ACTIVATION FUNCTIONS   #
############################

def _relu (x):
    '''
        REctified Linear Unit acivation function: relu(x) = max(0,x)
    '''
    return max(0,x)

def _identity (x):
    '''
        identity function: identity(x) = x
    '''
    return x

def _threshold (x):
    '''
        threshold activation function: treshold(x) = 1 if x>0
                                       treshold(x) = 0 if x<=0
    '''
    return 1 if x>0 else 0

def _logistic (x):
    '''
        logistic activation function: logistic(x) = 1 / (1 + exp(-x))
    '''
    return 1 / ( 1 + math.exp(-x) )

def _tanh (x):
    '''
        tanh activation function (returns hyperbolic tangent of the input) = tanh(x)
    '''
    return math.tanh (x)

def _zero_one_tanh (x):
    '''
        tanh activation function which output is from zero to one: _zero_one_tanh(x) = (1 + tanh(x))/2
    '''
    return (1 + math.tanh (x))/2


relu = np.vectorize (_relu, otypes=[float])
identity = np.vectorize (_identity, otypes=[float])
threshold = np.vectorize (_threshold, otypes=[float])
logistic = np.vectorize (_logistic, otypes=[float])
tanh = np.vectorize (_tanh, otypes=[float])
zero_one_tanh = np.vectorize (_zero_one_tanh, otypes=[float])

activation_functions = {
    "relu": relu,
    "identity": identity,
    "threshold": threshold,
    "logistic": logistic,
    "tanh": tanh,
    "zero_one_tanh": zero_one_tanh,
}


########################################
#   ACTIVATION FUNCTIONS DERIVATIVES   #
########################################

def _relu_derivative (x):
    '''
        REctified Linear Unit activation function derivative: relu'(x) = 0 if x<0
                                                              relu'(x) = 1 if x>=0
    '''
    return 0 if x<=0 else 1

def _identity_derivative (x):
    '''
        identity function derivative: identity'(x) = 1
    '''
    return 1

def _threshold_derivative (x):
    '''
        threshold activation function derivative: treshold'(x) = 0
    '''
    return 0

def _logistic_derivative (x):
    '''
        logistic activation function derivative: logistic'(x) = logistic(x) * ( 1 - logistic(x) )
    '''
    return _logistic(x) * ( 1 - _logistic (x) )

def _tanh_derivative (x):
    '''
        tanh activation function derivatives: tanh'(x) = 1 - (tanh(x))**2
    '''
    return 1 - (math.tanh (x))**2

def _zero_one_tanh_derivative (x):
    '''
        zero-one tanh activation function derivatives: tanh'(x) = 1/2 * ( 1 - (tanh(x))**2 )
    '''
    return 1/2 * ( 1 - (math.tanh (x))**2 )

relu_derivative = np.vectorize (_relu_derivative, otypes=[float])
identity_derivative = np.vectorize (_identity_derivative, otypes=[float])
threshold_derivative = np.vectorize (_threshold_derivative, otypes=[float])
logistic_derivative = np.vectorize (_logistic_derivative, otypes=[float])
tanh_derivative = np.vectorize (_tanh_derivative, otypes=[float])
zero_one_tanh_derivative = np.vectorize (_zero_one_tanh_derivative, otypes=[float])

activation_functions_derivatives = {
    "relu": relu_derivative,
    "identity": identity_derivative,
    "threshold": threshold_derivative,
    "logistic": logistic_derivative,
    "tanh": tanh_derivative,
    "zero_one_tanh": zero_one_tanh_derivative
}

######################
#   LOSS FUNCTIONS   #
######################

def _squaredLoss ( true_output, predicted_output ):
    '''
        squared loss: squaredLoss (t, p) = 1/2 * (t - p)^2 
    '''
    return 1/2 * (true_output - predicted_output)**2

def _binaryLogLoss ( true_output, predicted_output ):
    '''
        loss function for binary classification task: binaryLogLoss(t,p) = - (t * log(p) + (1-t) * log(1-p))
        t ∈ {0,1} is the true class.
        p ∈ [0,1] is the predicted probability for class 1.

        to compute this loss p is restricted in [1e-6, 1-1e-6] to avoid the extreme values of the logrithm function.
    '''
    eps = 1e-6
    p = np.clip (predicted_output, eps, 1-eps)
    return - (true_output * math.log (p) + (1-true_output) * math.log (1-p))

squaredLoss = np.vectorize (_squaredLoss, otypes=[float])
binaryLogLoss = np.vectorize (_binaryLogLoss, otypes=[float])

loss_functions = {
    "squared": squaredLoss,
    "log_loss": binaryLogLoss
}

##################################
#   LOSS FUNCTIONS DERIVATIVES   #
##################################

def _squaredLoss_derivative ( true_output, predicted_output ):
    '''
        squared loss derivative wrt predicted output p: squaredLoss' (t, p) = -(t - p)
    '''
    return predicted_output - true_output

def _binaryLogLoss_derivative ( true_output, predicted_output ):
    '''
        derivative of the loss function for binary classification task wrt predicted output p: 
        binaryLogLoss'(t,p) = - (t/p - (1-t)/(1-p))
        
        t ∈ {0,1} is the true class.
        p ∈ [0,1] is the predicted probability for class 1.

        to compute this loss p is restricted in [1e-6, 1-1e-6] to avoid the extreme values of the logrithm function.
    '''
    eps = 1e-6
    p = np.clip (predicted_output, eps, 1-eps)
    return (1 - true_output) / (1 - p) - true_output/p

squaredLoss_derivative = np.vectorize ( _squaredLoss_derivative, otypes=[float] )
binaryLogLoss_derivative = np.vectorize ( _binaryLogLoss_derivative, otypes=[float] )

loss_functions_derivatives = {
    "squared": squaredLoss_derivative,
    "log_loss": binaryLogLoss_derivative
}

##########################
#   ACCURACY FUNCTIONS   #
##########################

def _euclidean_loss (true_output, predicted_output):
    squares = (true_output - predicted_output) ** 2
    sum_of_squares = np.sum (squares, axis=1)
    distances = np.sqrt (sum_of_squares)
    return np.average (distances)

def _classification_loss (true_output, predicted_output):
    return sum (map (lambda x:1 if x else 0, true_output.ravel() != predicted_output.ravel() ) ) / len(true_output)

accuracy_functions = {
    "euclidean": _euclidean_loss,
    "classification": _classification_loss
}
