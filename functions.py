
import numpy as np
import math

############################
#   ACTIVATION FUNCTIONS   #
############################

def _relu (x):
    '''
        REctified Linear Unit output function: relu(x) = max(0,x)
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

relu = np.vectorize (_relu, otypes=[float])
identity = np.vectorize (_identity, otypes=[float])
threshold = np.vectorize (_threshold, otypes=[float])
logistic = np.vectorize (_logistic, otypes=[float])

activation_functions = {
    "relu": relu,
    "identity": identity,
    "threshold": threshold,
    "logistic": logistic
}


########################################
#   ACTIVATION FUNCTIONS DERIVATIVES   #
########################################

def _relu_derivative (x):
    '''
        REctified Linear Unit output function derivative: relu'(x) = 0 if x<0
                                                          relu'(x) = 1 if x>=0
    '''
    return 0 if x<0 else 1

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

relu_derivative = np.vectorize (_relu_derivative, otypes=[float])
identity_derivative = np.vectorize (_identity_derivative, otypes=[float])
threshold_derivative = np.vectorize (_threshold_derivative, otypes=[float])
logistic_derivative = np.vectorize (_logistic_derivative, otypes=[float])

activation_functions_derivatives = {
    "relu": relu_derivative,
    "identity": identity_derivative,
    "threshold": threshold_derivative,
    "logistic": logistic_derivative
}

######################
#   LOSS FUNCTIONS   #
######################


def _squaredLoss ( true_output, predicted_output ):
    '''
        squared loss: squaredLoss (t, p) = 1/2 * (t - p)^2 
    '''
    return 1/2 * (true_output - predicted_output)**2

squaredLoss = np.vectorize (_squaredLoss, otypes=[float])

loss_functions = {
    "squared": squaredLoss
}

##################################
#   LOSS FUNCTIONS DERIVATIVES   #
##################################

def _squaredLoss_derivative ( true_output, predicted_output ):
    '''
        squared loss derivative wrt predicted output p: squaredLoss' (t, p) = -(t - p)
    '''
    return predicted_output - true_output

squaredLoss_derivative = np.vectorize ( _squaredLoss_derivative, otypes=[float] )

loss_functions_derivatives = {
    "squared": squaredLoss_derivative
}
