
import numpy as np
import math

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
        threshold activation function: treshold(x) = x / |x|
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