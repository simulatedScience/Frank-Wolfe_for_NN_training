"""
this module conducts parameter studies on neural networks for a chemistry application.

Author: Sebastian Jost
"""
from dense_parameter_study import dense_parameter_study
from chem_data_prep import load_chem_data

###################
# Chem parameters #
###################

neural_net_params = {
    "neurons_per_layer"       : [(256, 128), (64, 64)],
    "input_shape"             : 36,
    "output_shape"            : 1,
    "activation_functions"    : "relu",
    "last_activation_function": "linear",
    "layer_types"             : "dense",
    "loss_function"           : "mean_squared_error",
    }
training_params = {
    "training_data_percentage": 1,
    "number_of_epochs"     : 50,
    "batch_size"           : 100,
    "validation_split"     : 0.2,
    "number_of_repetitions": 3,
    }

optimizer_params = [
    {
        "optimizer_type": "Adam",
        "learning_rate": 0.01,
        "epsilon": 1e-7,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "weight_decay": 0.01,
    },
    {
        "optimizer_type": "SGD",
        "learning_rate": 0.01,
        "weight_decay": 0.01,
    },
    # {
    #     "optimizer_type": "mSFW",
    #     "learning_rate": 0.01,
    #     "momentum": 0.9,
    #     "constraints_type": "unconstrained",
    #     "rescale": None,
    # },
    # {
    #     "optimizer_type": "mSFW",
    #     "learning_rate": 0.01,
    #     "momentum": 0.9,
    #     "constraints_type": "L1",
    #     "constraints_radius": 300,
    #     "rescale": None,
    # },
    {
        "optimizer_type": "mSFW",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "constraints_type": "L2",
        "constraints_radius": 300,
        "rescale": None,
    },
    {
        "optimizer_type": "mSFW",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "constraints_type": "hypercube",
        "constraints_radius": 300,
        "rescale": None,
    },
    # {
    #     "optimizer_type": "mSFW",
    #     "learning_rate": 0.01,
    #     "momentum": 0.9,
    #     "constraints_type": "ksparse",
    #     "constraints_radius": 300,
    #     "constraints_K": 1000,
    #     "rescale": None,
    # },
    {
        "optimizer_type": "mSFW",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "constraints_type": "ksparse",
        "constraints_radius": 300,
        "constraints_K": 2000,
        "rescale": None,
    },
    # {
    #     "optimizer_type": "mSFW",
    #     "learning_rate": 0.01,
    #     "momentum": 0.9,
    #     "constraints_type": "knorm",
    #     "constraints_radius": 300,
    #     "constraints_K": 1000,
    #     "rescale": None,
    # },
    {
        "optimizer_type": "mSFW",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "constraints_type": "knorm",
        "constraints_radius": 300,
        "constraints_K": 2000,
        "rescale": None,
    },
]
dataset = load_chem_data()
dense_parameter_study(neural_net_params, training_params, optimizer_params, dataset)