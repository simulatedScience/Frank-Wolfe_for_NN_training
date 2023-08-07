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
# from _constraints import LpBall, KSparsePolytope, KNormBall, Unconstrained, Constraint
from _constraints import Constraint, create_lp_constraints, create_unconstraints, create_k_sparse_constraints, create_k_norm_constraints

class NeuralNet(nn.Module):
    """
    A Class to build Pytorch neural networks
    """
    def __init__(self,
                 model_params: dict[str, int | tuple | list | str],
                 loss_function,
                 track_accuracy = "auto"):
        super(NeuralNet, self).__init__()
        self.model_params = model_params
        self.loss_function = loss_function
        # initialize history
        self.history: dict[str, list] = {"loss": [], "val_loss": []}
        if track_accuracy == "auto":
            self.track_accuracy: bool = True if model_params["output_shape"] > 1 else False
        else:
            self.track_accuracy: bool = track_accuracy
        if self.track_accuracy:
            self.history["accuracy"] = []
            self.history["val_accuracy"] = []
            
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

    def init_testing(self):
        """
        Initialize the network for testing
        """
        self.network.eval()
        self.history["test_loss"] = []
        if self.track_accuracy:
            self.history["test_accuracy"] = []
        

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

    def set_constraints(self, constraints: list[Constraint]):
        """
        Set the constraints for the optimizer

        Args:
            constraints (list[Constraint]): list of constraints
        """
        self.constraints = constraints

    def set_optimizer(self, optimizer: optim.Optimizer):
        """
        Set the optimizer for the model

        Args:
            optimizer (torch.nn.optim.Optimizer): optimizer
        """
        self.optimizer = optimizer
        
    @torch.no_grad()
    def evaluate(self, x_val, y_val, batch_size=32, device="cpu", test_mode=False):
        """
        Evaluate the model on the validation data

        Args:
            x_val (torch.Tensor): input data
            y_val (torch.Tensor): target data
            batch_size (int, optional): batch size. Defaults to 32.

        Returns:
            tuple: (loss, output)
        """
        val_loss = 0
        val_output = []
        for i in range(0, len(x_val), batch_size):
            x_batch = x_val[i:i+batch_size]
            y_batch = y_val[i:i+batch_size]
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = self.forward(x_batch)
            val_loss += self.loss_function(output, y_batch).item()
            val_output.append(output)
        val_output = torch.cat(val_output)
        if test_mode:
            prefix = "test_"
        else:
            prefix = "val_"
        self.history[prefix + "loss"].append(val_loss)
        if self.track_accuracy:
            self.history[prefix + "accuracy"].append(calculate_accuracy(val_output, y_val, device=device))
        return val_loss, val_output

    def fit_batch(self, x_data, y_data, constraints=None, device="cpu"):
        """
        Fit the model to the given data

        Args:
            x_data (torch.Tensor): input data
            y_data (torch.Tensor): target data
            constraints (list[torch.Tensor], optional): list of constraints. Defaults to None.
        
        Returns:
        """
        output = self.forward(x_data)
        loss = self.loss_function(output, y_data)
        self.optimizer.zero_grad()
        loss.backward()
        if constraints is None:
            self.optimizer.step()
        else:
            self.optimizer.step(constraints=constraints)
        if self.track_accuracy:
            self.history["accuracy"].append(calculate_accuracy(output, y_data, device=device))
        self.history["loss"].append(loss.item())
        return loss.item(), output
    
    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int,
        batch_size: int,
        validation_split: float,
        device: str = "cpu",
        verbose: int = 0
    ):
        """
        train the model with the given data

        Args:
            x_train (torch.Tensor): input data
            y_train (torch.Tensor): labels
            model (NeuralNet): neural network model
            epochs (int): number of epochs
            batch_size (int): batch size
            validation_split (float): validation split
            verbose (int, optional): verbosity. Defaults to 0.
        """
        self.to(device)
        self.loss_function.to(device)
        self.train() # set model to training mode (enables dropout etc.)
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
                self.fit_batch(x_batch, y_batch, constraints=self.constraints, device=device)
            # evaluate model
            if x_validation is not None:
                self.evaluate(x_validation, y_validation, batch_size=batch_size, device=device)
            # print progress
            if verbose > 0:
                epoch_progress: str = f"finished epoch {epoch+1}/{epochs}"
        
        return self.history

def calculate_accuracy(output, y_data, device="cpu"):
    if output.device != y_data.device:
        output = output.to(device)
        y_data = y_data.to(device)
    _, predicted = torch.max(output, 1)
    if len(y_data.shape) > 1: # convert one-hot encoding to labels
        _, y_data = torch.max(y_data, 1)
    correct = (predicted == y_data).sum().item()
    accuracy = correct / len(y_data)
    return accuracy

def set_optimizer(
        model: NeuralNet,
        **params: dict) -> torch.optim.Optimizer:
    """
    Convert a dict of optimizer parameters to a torch optimizer

    Args:
        model (NeuralNet): neural network model
        optimizer (str | torch.optim.Optimizer): optimizer
    """
    optimizer = params["optimizer_type"]
    torch_optimizer = None
    if not isinstance(optimizer, str):
        torch_optimizer = optimizer
    if optimizer.lower() == "adam": # Adam
        torch_optimizer = optim.Adam(
            model.parameters(),
            lr=params["learning_rate"],
            eps=params["epsilon"],
            betas=(params.get("beta_1", 0.9), params.get("beta_2", 0.999)),
            weight_decay=params.get("weight_decay", 0),
        )
    elif optimizer.lower() == "sgd": # Stochastic Gradient Descent
        torch_optimizer = optim.SGD(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params.get("weight_decay", 0),)
    elif optimizer.lower() == "msfw": # Stochastic Frank-Wolfe with momentum
        torch_optimizer = SFW(model.parameters(),
                   learning_rate=params["learning_rate"],
                   momentum=params.get("momentum", 0),
                   rescale=params.get("rescale", "gradient"),)
    elif optimizer.lower() == "csgd": # Stochastic Gradient Descent with constraints
        torch_optimizer = CSGD(model.parameters(),
                    learning_rate=params["learning_rate"],
                    momentum=params["momentum"],
                    dampening=params["dampening"],
                    weight_decay=params["weight_decay"],
                    nestrov=params["nestrov"])
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    return torch_optimizer

def get_constraints(
        model: NeuralNet,
        **params: dict) -> Constraint:
    """_summary_

    Args:
        model (NeuralNet): _description_
        optimizer (str): _description_

    Returns:
        _type_: _description_
    """
    optimizer = params["optimizer_type"]
    if optimizer.lower() != "msfw":
        constraints = None
    elif optimizer.lower() in ("msfw", "csgd"):
        if params["constraints_type"].lower() == "unconstrained":
            constraints = create_unconstraints(
                model=model,
            )
        elif params["constraints_type"].lower() in ("l1", "l_1"):
            constraints = create_lp_constraints(
                model=model,
                ord=1,
                mode="initialization",
                value=params["constraints_radius"],
            )
        elif params["constraints_type"].lower() in ("l2", "l_2"):
            constraints = create_lp_constraints(
                model=model,
                ord=2,
                mode="initialization",
                value=params["constraints_radius"],
            )
        elif params["constraints_type"].lower() in ("linf", "l_inf", "hypercube"):
            constraints = create_lp_constraints(
                model=model,
                ord=float("inf"),
                mode="initialization",
                value=params["constraints_radius"],
            )
        elif params["constraints_type"].lower() == "ksparse":
            constraints = create_k_sparse_constraints(
                model=model,
                K=params["constraints_K"],
                mode="initialization",
                value=params["constraints_radius"],
            )
        elif params["constraints_type"].lower() == "knorm":
            constraints = create_k_norm_constraints(
                model=model,
                K=params["constraints_K"],
                mode="initialization",
                value=params["constraints_radius"],
            )
    return constraints
    

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
    loss_function = set_loss_function(model_params)
    
    model = NeuralNet(model_params, loss_function)
    optimizer = set_optimizer(model, **optimizer_params)
    model.set_optimizer(optimizer)
    constraints = get_constraints(model, **optimizer_params)
    model.set_constraints(constraints)
    

    return model

