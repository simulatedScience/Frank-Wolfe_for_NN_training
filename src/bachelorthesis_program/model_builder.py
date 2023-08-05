"""
a module for building tensorflow neural network models

Author: Sebastian Jost
"""
# Updated PyTorch code
import numpy as np                  
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.utils as skutils

from _custom_optimizers import SFW # Stochastic Frank-Wolfe with momentum
from _custom_optimizers import SGD as CSGD # Stochastic Gradient Descent with constraints
from _constraints import LpBall, KSparsePolytope, KNormBall

class NeuralNet(nn.Module):
    """
    A Class to build Pytorch neural networks
    """
    def __init__(self,
                 model_params: dict[str, int | tuple | list | str],
                 optimizer: optim.Optimizer,
                 loss_function: nn.Loss):
        super(NeuralNet, self).__init__()
        self.model_params = model_params
        self.optimizer = optimizer
        self.loss_function = loss_function

        # Setting activations
        self.activations, self.last_activation = self.set_activations(
            model_params["activation_functions"],
            len(model_params["neurons_per_layer"]),
            model_params["last_activation_function"]
        )

        # Building the network
        layers = []
        input_size = model_params["input_shape"][0] * model_params["input_shape"][1]
        for idx, n_neurons in enumerate(model_params["neurons_per_layer"]):
            layers.append(nn.Linear(input_size, n_neurons))
            layers.append(self.get_activation(self.activations[idx]))
            input_size = n_neurons

        # Output layer
        layers.append(nn.Linear(input_size, model_params["output_shape"]))
        layers.append(self.get_activation(self.last_activation))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        Pass an input x through the network and get the NNs output

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.network(x)

    def set_activations(self,
                activations: list[str] | str,
                n_hidden_layers: int,
                last_activation: str = None):
        """
        Set the activation functions for the hidden layers and the output layer

        Args:
            activations (list[str] | str): list of `n_hidden_layers` activation functions or a single activation function for all hidden layers. If this is a string and no `last_activation` is given, the same activation function will be used for all hidden layers and the output layer. Otherwise, the last activation function will be used for the output layer.
            If a list is given, the last element of the list will be used as the activation function for the output layer.
            n_hidden_layers (int): number of hidden layers
            last_activation (str, optional): activation function for the output layer. This is only allowed if `activations` is a string. Defaults to None.

        Returns:
            tuple: (list of activation functions for the hidden layers, activation function for the output layer)

        Raises:
            ValueError: if the number of activation functions given as a list does not match the number of hidden layers
        """
        if isinstance(activations, str):
            if last_activation is None:
                return [activations]*n_hidden_layers, activations
            return [activations]*n_hidden_layers, last_activation

        if isinstance(activations, (list, tuple)):
            if len(activations) != n_hidden_layers:
                raise ValueError(f"Number of activation functions ({len(activations)}) does not match number of hidden layers ({n_hidden_layers})")
            return activations, activations[-1]

    def get_activation(self, activation_name):
        if activation_name.lower() == "relu":
            return nn.ReLU()
        elif activation_name.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation_name.lower() == "softmax":
            return nn.Softmax(dim=-1)
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")

    def fit_batch(self, x_data, y_data, constraints=None):
        """
        Fit the model to the given data

        Args:
            x_data (torch.Tensor): input data
            y_data (torch.Tensor): target data
            constraints (list[torch.Tensor], optional): list of constraints. Defaults to None.
        
        Returns:
        """
        self.train() # set model to training mode (enables dropout etc.)
        output = self.forward(x_data)
        loss = self.loss_function(output, y_data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step(constraints=constraints)
        return loss.item(), output
    
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        
        epochs: int,
        batch_size: int,
        validation_split: float,
        constraints: list = None,
        device: str = "cpu",
        verbose: int = 0
    ):
        """
        train the model with the given data

        Args:
            x_train (np.ndarray): input data
            y_train (np.ndarray): labels
            model (NeuralNet): neural network model
            epochs (int): number of epochs
            batch_size (int): batch size
            validation_split (float): validation split
            constraints (list, optional): list of constraints. Defaults to None.
            verbose (int, optional): verbosity. Defaults to 0.
        """
        # split data into training and validation data
        if validation_split > 0:
            n_samples = len(x_train)
            n_validation = round(n_samples*validation_split)
            x_train, x_validation = x_train[:-n_validation], x_train[-n_validation:]
            y_train, y_validation = y_train[:-n_validation], y_train[-n_validation:]
        else:
            x_validation, y_validation = None, None
        
        for epoch in range(epochs):
            # shuffle data
            x_train, y_train = skutils.shuffle(x_train, y_train)
            # train model
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                self.fit_batch(x_batch, y_batch, self.optimizer, self.loss_function, constraints)
            # evaluate model
            if x_validation is not None:
                self.evaluate(x_validation, y_validation, self.loss_function)
            # print progress
            if verbose > 0:
                epoch_progress: str = f"finished epoch {epoch+1}/{epochs}"


def set_optimizer_constrained(
        model: NeuralNet,
        optimizer: str | torch.optim.Optimizer,
        **params: dict):
    """
    Convert a dict of optimizer parameters to a torch optimizer and a constraints object

    Args:
        model (NeuralNet): neural network model
        optimizer (str | torch.optim.Optimizer): optimizer
        
    """
    constraints = params.get("constraints", None)
    torch_optimizer = None
    if not isinstance(optimizer, str):
        torch_optimizer = optimizer
    if optimizer.lower() == "adam": # Adam
        torch_optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"], eps=params["epsilon"],
                          betas=(params["beta_1"], params["beta_2"]))
    elif optimizer.lower() == "sgd": # Stochastic Gradient Descent
        torch_optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"])
    elif optimizer.lower() == "msfw": # Stochastic Frank-Wolfe with momentum
        torch_optimizer = SFW(model.parameters(),
                   lr=params["learning_rate"],
                   momentum=params["momentum"],
                   rescale=params["rescale"])
    elif optimizer.lower() == "csgd": # Stochastic Gradient Descent with constraints
        torch_optimizer = CSGD(model.parameters(),
                    lr=params["learning_rate"],
                    momentum=params["momentum"],
                    dampening=params["dampening"],
                    weight_decay=params["weight_decay"],
                    nestrov=params["nestrov"])
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    
    constraints = get_constraints(optimizer, **params)
    return torch_optimizer, constraints

def get_constraints(
    optimizer: str,
    **params: dict):
    if optimizer.lower() != "msfw":
        constraints = None
    elif optimizer.lower() == "msfw":
        # number of dimensions = umber of model parameters
        n = sum(p.numel() for p in model.parameters())
        if params["constraints_type"].lower == "l1":
            constraints = LpBall(
                n=n,
                ord=1,
                radius=params["constraints_radius"],)
        elif params["constraints_type"].lower == "l2":
            constraints = LpBall(
                n=n,
                ord=2,
                radius=params["constraints_radius"],)
        elif params["constraints_type"].lower in ("linf", "hypercube"):
            constraints = LpBall(
                n=n,
                ord=float("inf"),
                radius=params["constraints_radius"],)
        elif params["constraints_type"].lower == "ksparse":
            constraints = KSparsePolytope(
                n=n,
                K=params["constraints_K"],
                radius=params["constraints_radius"],)
        elif params["constraints_type"].lower() == "knorm":
            constraints = KNormBall(
                n=n,
                K=params["constraints_K"],
                radius=params["constraints_radius"],)
        
    

def set_loss_function(model_params):
    # Mapping the loss function to PyTorch equivalent
    if model_params["loss_function"].lower() == "categorical_crossentropy":
        loss_function = nn.CrossEntropyLoss()
    elif model_params["loss_function"].lower() in ("mse", "mean_squared_error"):
        loss_function = nn.MSELoss()
    elif model_params["loss_function"].lower() in ("mae", "mean_absolute_error"):
        loss_function = nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {model_params['loss_function']}")
    return loss_function

def make_model(model_params, optimizer_params):
    optimizer, constraints = set_optimizer_constrained(model, **optimizer_params)
    loss_function = set_loss_function(model_params)
    model = NeuralNet(model_params, optimizer, loss_function, constraints)
    

    return model, optimizer, loss_function
