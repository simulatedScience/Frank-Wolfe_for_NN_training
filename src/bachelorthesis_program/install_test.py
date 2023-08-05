"""
this module implements a small example for a parameter study to test functionality quickly.

Author: Sebastian Jost
"""
from dense_parameter_study import dense_parameter_study
from mnist_data_prep import load_mnist

neural_net_params_debug = {
    "neurons_per_layer"       : (50,10),
    "input_shape"             : (28,28),
    "output_shape"            : 10,
    "activation_functions"    : "relu",
    "last_activation_function": "sigmoid",
    "layer_types"             : "dense",
    "loss_function"           : "categorical_crossentropy"
    }
training_params_debug = {
    "training_data_percentage": 1,
    "number_of_epochs"     : 50,
    "batch_size"           : 1000,
    "validation_split"     : 0.2,
    "number_of_repetitions": 5
    }
optimizer_params_debug = {
    "optimizer"    : "adam",
    "learning_rate": 0.001,
    "epsilon"      : 1e-7,
    "beta_1"       : 0.9,
    "beta_2"       : 0.999
    }
mnist_data = load_mnist()
dense_parameter_study(
        neural_net_params_debug,
        training_params_debug,
        optimizer_params_debug,
        dataset=mnist_data)