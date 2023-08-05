"""
this module provides functions to convert loss computed with some loss function to the corresponding mean absolute arror.
however this only works for single-node outputs and is only implemented for logcosh and mean squared error.

Author: Sebastian Jost
"""
import numpy as np

def loss_conversion(loss_value, loss_function):
    if loss_function == "mean_squared_error":
        return mse_inverse(loss_value)
    if loss_function == "log_cosh":
        return logcosh_inverse(loss_value)

def mse_inverse(x):
    """
    Convert a loss value computed with 'mean squared error' to a 'mean absolute error' value.
    This only works for single-node outputs.
    """
    return np.sqrt(x)


def logcosh_inverse(x):
    """
    Convert a loss value computed with 'log cosh' to a 'mean absolute error' value.
    This only works for single-node outputs.
    """
    return np.log(np.sqrt(np.exp(2*x) - 1) + np.exp(x))