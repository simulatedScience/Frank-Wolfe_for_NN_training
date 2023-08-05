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
    "neurons_per_layer"       : [(40, 20), (50, 10), (30, 30, 10)],
    "input_shape"             : 36,
    "output_shape"            : 1,
    "activation_functions"    : ["relu", "sigmoid"],
    "last_activation_function": "linear",
    "layer_types"             : "dense",
    "loss_function"           : ["mean_squared_error", "log_cosh"]
    }
training_params = {
    "training_data_percentage": 1,
    "number_of_epochs"     : 500,
    "batch_size"           : [100, 1000, 10_000],
    "validation_split"     : 0.2,
    "number_of_repetitions": 1
    }
optimizer_params = {
    "optimizer"    : ["adam", "c_adam"],
    "learning_rate": [0.01, 0.001],
    "epsilon"      : 1e-7,
    "beta_1"       : 0.9,
    "beta_2"       : 0.999
    }
dataset = load_chem_data()
print(len(dataset[1][1])+len(dataset[0][1]))
dense_parameter_study(neural_net_params, training_params, optimizer_params, dataset)