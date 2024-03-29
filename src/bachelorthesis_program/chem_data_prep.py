import os

import sklearn.utils as skutils
import torch

from chem_data.parse_chem_data import parse_chem_data


def load_chem_data(test_samples=None, test_percentage=None):
    """
    load chemistry data in shape (x_train, y_train), (x_test, y_test)
    number of test samples can be specified as number of samples or percentage of availiable data
        (rounded to the nearest integer)
    """
    if test_samples is None and test_percentage is None:
        test_percentage = 0.1
    inputs, outputs = parse_chem_data(os.path.join(os.path.dirname(__file__), "chem_data", "features_augmented_05percent.arff"))
    return test_split(inputs, outputs, test_samples=test_samples, test_percentage=test_percentage)

def test_split(inputs, outputs, test_samples=None, test_percentage=None):
    """
    split training data for neural networks into a training and testing dataset.
    data should be given as inputs and outputs and will be returned as two input-output pairs.
    number of test samples can be specified as number of samples or percentage of availiable data
        (rounded to the nearest integer)
    """
    shuffled_in, shuffled_out = skutils.shuffle(inputs, outputs) # shuffle both sets identically
    if test_percentage is not None:
        test_samples = round(len(inputs)*test_percentage)
    x_train = torch.tensor(shuffled_in[test_samples:], dtype=torch.float32)
    y_train = torch.tensor(shuffled_out[test_samples:], dtype=torch.float32)
    y_train = y_train.reshape((y_train.shape[0], 1))
    x_test  = torch.tensor(shuffled_in[:test_samples], dtype=torch.float32)
    y_test  = torch.tensor(shuffled_out[:test_samples], dtype=torch.float32)
    y_test = y_test.reshape((y_test.shape[0], 1))
    return (x_train, y_train), (x_test, y_test)