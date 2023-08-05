"""
Data loading and preprocessing for MNIST dataset.

Author: Sebastian Jost
"""

import numpy as np
import tensorflow.keras as keras


def load_mnist():
    """
    load and prepare mnist data
    use one-hot encoded outputs and normalized inputs

    returns:
    --------
        (x_train, y_train), (x_test, y_test), where:
            x_train - 60000x28x28 array with float values in range [0,1] - training images
            y_train - 60000x10 array with integer labels in range [0,1] - training labels
            x_test - 10000x28x28 array with float values in range [0,1] - test images
            y_test - 10000x10 array with integer labels in range [0,1] - test labels
    """
    all_data = keras.datasets.mnist.load_data()
    return prepare_mnist_data(all_data)


def load_raw_mnist():
    """
    load mnist dataset from keras

    (x_train, y_train), (x_test, y_test) = load_raw_mnist()

    returns:
    --------
        x_train - 60000x28x28x1 array with integer values in range [0,255] - training images
        y_train - 60000x1 array with integer labels in range [0,9] - training labels
        x_test - 10000x28x28x1 array with integer values in range [0,255] - test images
        y_test - 10000x1 array with integer labels in range [0,9] - test labels
    """
    all_data = keras.datasets.mnist.load_data()
    return all_data

def one_hot_encode(labels, n_nodes=10):
    """
    one-hot encode labels given as consecutive integers starting at 0

    input:
    ------
        labels - (array-like) of (int) - list, tuple or array of consecutive integers starting at 0

    returns:
    --------
        (np.ndarray) of (int) - 2d array of integers 0 and 1.
            The i-th row represents the one-hot encoded label of the i-th entry in `labels`
    """
    new_data = np.zeros((labels.size, n_nodes))
    new_data[np.arange(labels.size), labels] = 1
    return new_data


def prepare_mnist_data(all_data):
    (x_train, y_train), (x_test, y_test) = all_data
    
    print("Number of original training examples:", len(x_train))
    print("Number of original test examples:", len(x_test))
    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
    # reshape x_train
    x_shape = x_train.shape
    x_train = x_train.reshape((x_shape[0], x_shape[1], x_shape[2]))
    # reshape x_test
    x_shape = x_test.shape
    x_test = x_test.reshape((x_shape[0], x_shape[1], x_shape[2]))
    # make labels one-hot encoded
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)
    return (x_train, y_train), (x_test, y_test)
