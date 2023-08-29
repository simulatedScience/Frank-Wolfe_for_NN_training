"""
This module provides functions for training neural networks.

Author: Sebastian Jost
"""
import time

import sklearn.utils as skutils
import numpy as np
import torch
# from torch.optim import Optimizer
# from torch.nn import Loss

from model_builder import make_model, NeuralNet
from file_management import save_model
from pickleable_history import get_save_history
from helper_functions import get_date_time, auto_use_multiprocessing


def train_model(model_params, optimizer_params, training_data,
                sub_folders=None, verbose=2):
    """
    create, train and save a model

    inputs:
    -------
        model_params - (dict) - dict containing all necessary information about network layout and training parameters.
            specifically the following keys should be included:
            # model parameters
            - neurons_per_layer - (tuple) of (int)
            - input_shape - (int) or (tuple) of (int)
            - output_shape - (int)
            - activation_functions - (str) or (tuple) of (str)
            - last_activation_function - (str)
            - layer_types - (str)
            - loss_function - (str)
            # training parameters
            - train_data_percentage - (float)
            - maximum_training_time - (int)
            - batch_size - (int)
            - validation_split - (tuple) of (float)
        optimizer_params - (dict) - dict containing all necessary information about the optimizer to be used.
            specifically the following keys should be included:
            # optimizer parameters
            - optimizer - (str)
            - learning_rate - (float)
            - epsilon - (float)
            - beta_1 - (float)
            - beta_2 - (float)

        training_data - (tuple) - tuple containing input data and labels each as a list or array

    returns:
    --------
        model - (tf.model) - tensorflow model object of the trained neural network
        history - (pickleable_history) - pickleable history object containing info about the training process and performance
    """
    model = make_model(model_params, optimizer_params)

    x_train, y_train = choose_training_data(
        training_data,
        model_params["training_data_percentage"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # use_multiprocessing = auto_use_multiprocessing()
    start_time = time.time()
    history = model.fit(
            x_train,
            y_train,
            epochs=model_params["number_of_epochs"],
            batch_size=model_params["batch_size"],
            validation_split=model_params["validation_split"],
            device=device,
            verbose=0)
    training_time = time.time()-start_time
    # TODO save model
    save_time = get_date_time()
    model_path = save_model(model, save_time=save_time,
                            sub_folders=sub_folders)
    history = get_save_history(
        history, model_params, optimizer_params, model_path, training_time, save_time)
    return model, history


def choose_training_data(data: tuple[np.ndarray], data_percentage: float = 1):
    """
    randomly choose `data_percentage*100` percent from the given data.

    inputs:
    -------
        data - (list) of (numpy arrays) - data given as tuple or list containing a list of inputs and a list of labels
        data_percentage - (float) in range ]0,1] - for choosing random subsets of the data
    """
    n_samples = round(len(data[0])*data_percentage)
    return skutils.shuffle(*data, n_samples=n_samples)



if __name__ == "__main__":
    # print("use multiprocessing:", auto_use_multiprocessing())
    model_params = {
        "neurons_per_layer"       : (64, 16),
        "input_shape"             : 36,
        "output_shape"            : 1,
        "activation_functions"    : "relu",
        "last_activation_function": "linear",
        "layer_types"             : "dense",
        "loss_function"           : "mean_squared_error",
        "training_data_percentage": 1,
        "number_of_epochs"     : 50,
        "batch_size"           : 100,
        "validation_split"     : 0.2,
        "number_of_repetitions": 5,
    }
    optimizer_params = {
        "optimizer_type": "Adam",
        "learning_rate": 0.01,
        "epsilon": 1e-7,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "weight_decay": 0.01,
    }
    from chem_data_prep import load_chem_data
    dataset = load_chem_data()
    x_train, y_train = dataset[0]
    filepath = "test_adam_training"
    trained_model, history = train_model(
        model_params=model_params,
        optimizer_params=optimizer_params,
        training_data=(x_train, y_train),
        sub_folders=filepath,
        verbose=2
    )
    from file_management import save_picklable_object
    save_time = time.time()
    save_picklable_object(
        history,
        filename="history.pickle",
        save_time=history.save_time,
        sub_folders=filepath)
    print(history)
