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
    "neurons_per_layer": (32,),
    "input_shape": (28, 28),
    "output_shape": 10,
    "activation_functions": "relu",
    "last_activation_function": "softmax",
    "layer_types": "dense",
    "loss_function": "categorical_crossentropy",
}
training_params = {
    "training_data_percentage": 1,
    "number_of_epochs": 20,
    "batch_size": 100,
    "validation_split": 0.2,
    "number_of_repetitions": 5
}
optimizer_params = [
    {
        "optimizer_type": "Adam",
        "learning_rate": 0.01,
        "epsilon": 1e-7,
        "beta_1": 0.9,
        "beta_2": 0.999,
    },
    {
        "optimizer_type": "SGD",
        "learning_rate": 0.01,
    },
    {
        "optimizer_type": "mSFW",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "constraints_type": "unconstrained",
    },
    {
        "optimizer_type": "mSFW",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "constraints_type": "L1",
        "constraints_radius": 300,
    },
    {
        "optimizer_type": "mSFW",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "constraints_type": "L2",
        "constraints_radius": 300,
    },
    {
        "optimizer_type": "mSFW",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "constraints_type": "hypercube",
        "constraints_radius": 300,
    },
    {
        "optimizer_type": "mSFW",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "constraints_type": "ksparse",
        "constraints_radius": 300,
        "constraints_K": 1000,
    },
    {
        "optimizer_type": "mSFW",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "constraints_type": "ksparse",
        "constraints_radius": 300,
        "constraints_K": 2500,
    },
    {
        "optimizer_type": "mSFW",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "constraints_type": "knorm",
        "constraints_radius": 300,
        "constraints_K": 1000,
    },
    {
        "optimizer_type": "mSFW",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "constraints_type": "knorm",
        "constraints_radius": 300,
        "constraints_K": 2500,
    },
]
# {
#     "optimizer": ["adam", "c_adam"],
#     "learning_rate": [0.1, 0.01, 0.001],
#     "epsilon": [1, 1e-7],
#     "beta_1": 0.9,
#     "beta_2": 0.999
# }

dataset = load_mnist(one_hot=False)
dense_parameter_study(neural_net_params, training_params,
                      optimizer_params, dataset)
