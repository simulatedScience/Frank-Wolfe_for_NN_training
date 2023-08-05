"""
this module conducts parameter studies on neural networks for the MNIST dataset

Author: Sebastian Jost
"""

from dense_parameter_study import dense_parameter_study
from mnist_data_prep import load_mnist

####################
# MNIST parameters #
####################

neural_net_params = {
    "neurons_per_layer": (50, 10),
    "input_shape": (28, 28),
    "output_shape": 10,
    "activation_functions": "relu",
    "last_activation_function": ["softmax", "sigmoid"],
    "layer_types": "dense",
    "loss_function": "categorical_crossentropy"
}
training_params = {
    "training_data_percentage": 1.,
    "number_of_epochs": 50,
    "batch_size": 100,
    "validation_split": 0.2,
    "number_of_repetitions": 5
}
optimizer_params = {
    "optimizer": ["adam", "my_adam", "c_adam", "c_adam_hat"],
    "learning_rate": 0.001,
    "epsilon": 1e-7,
    "beta_1": 0.9,
    "beta_2": 0.999
}

dataset = load_mnist()
dense_parameter_study(neural_net_params, training_params,
                      optimizer_params, dataset)
