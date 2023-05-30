# Import necessary modules
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms

from torchmetrics import Accuracy, F1Score


class Dense_NN(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layers):
        super(Dense_NN, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_layers = hidden_layers

        layers = []
        input_size = input_shape[0] * input_shape[1]
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, output_shape[0]))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.layers(x)
        return x

class Trainable_Model(pl.LightningModule):
    def __init__(self,
            model,
            loss_fn: nn.Module = nn.CrossEntropyLoss(),
            optimizer: optim.Optimizer = optim.Adam,
            learning_rate: float = 0.001):
        super().__init__()
        self.model: nn.Module = model
        self.loss_fn: nn.Module = loss_fn
        self.optimizer: optim.Optimizer = optimizer
        # set up metrics
        # these keep an internal state
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.train_f1 = F1Score(num_classes=10)
        self.val_f1 = F1Score(num_classes=10)
        self.test_f1 = F1Score(num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the forward pass of the model

        Args:
            x (torch.Tensor): Input tensor to the model

        Returns:
            torch.Tensor: Output tensor from the model
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Implements forward pass for training data and logs metrics.
        
        Args:
            batch (torch.Tensor): Input data
            batch_idx (int): Index of the batch
            
        Returns:
            torch.Tensor: Loss value
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc(y_hat, y))
        self.log('train_f1', self.train_f1(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Implements forward pass for validation data and logs metrics.

        Args:
            batch (torch.Tensor): Input data
            batch_idx (int): Index of the batch
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc(y_hat, y))
        self.log('val_f1', self.val_f1(y_hat, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Implements forward pass for test data and logs metrics.
        
        Args:
            batch (torch.Tensor): Input data
            batch_idx (int): Index of the batch
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc(y_hat, y))
        self.log('test_f1', self.test_f1(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=0.001)

def load_mnist_data(
        batch_size=128,
        validation_split=0.2,
        num_workers=4,
        seed=5,
        verbosity=0):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_dataset = datasets.MNIST('../datasets', train=True, download=True, transform=transform)
    # Creating data indices for training and validation splits:
    dataset_size = len(mnist_dataset)
    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(seed))
    split = int(validation_split * dataset_size)
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(mnist_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    validation_loader = DataLoader(mnist_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    # Load the test set
    test_dataset = datasets.MNIST('../datasets', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if verbosity > 0:
        print("Successfully created dataloaders.")
    return train_loader, validation_loader, test_loader

def train_model(model, data_loader, validation_loader):
    lit_model = Trainable_Model(model, nn.CrossEntropyLoss(), optim.Adam)
    trainer = pl.Trainer(max_epochs=2, logger=False, gpus=1)
    trainer.fit(lit_model,
                train_dataloaders=data_loader,
                val_dataloaders=validation_loader)
    return trainer

def evaluate_model(model, data_loader, trainer):
    lit_model = Trainable_Model(model, nn.CrossEntropyLoss(), optim.Adam)
    # trainer = pl.Trainer(logger=False, gpus=1)
    trainer.test(lit_model, data_loader)
    return trainer

def print_results(trainer):
    print("results:")
    print(trainer.logged_metrics)
    
    print("Final Training Metrics: ", trainer.logged_metrics)
    print("Final Validation Metrics: ", trainer.callback_metrics)
    print("Final Test Metrics: ", trainer.logged_metrics)

def main():
    train_loader, validation_loader, test_loader = load_mnist_data(verbosity=1, num_workers=0)
    # model = Dense_NN((28, 28), (10,), tuple())
    # model = Dense_NN((28, 28), (10,), (50, 10))
    model = Dense_NN((28, 28), (10,), (200, 200))
    trainer = train_model(model, train_loader, validation_loader)
    trainer = evaluate_model(model, test_loader, trainer)
    print_results(trainer)

# def train(
#         model: nn.Module,
#         train_loader: DataLoader,
#         optimizer: optim.Optimizer,
#         loss_fn,
#         epochs: int = 5):
#     # Train the model
#     for epoch in range(epochs): 
#         # create progress bar to show progress of epoch
#         loop = tqdm(train_loader, total=len(train_loader), leave=True, ncols=100)
#         for images, labels in loop:
#             optimizer.zero_grad()
#             output = model(images)
#             loss = loss_fn(output, labels)
#             loss.backward()
#             optimizer.step()

#             # update progress bar
#             loop.set_description(f"Epoch [{epoch}/{epochs}]")
#             loop.set_postfix(loss = f"{loss.item():.3f}")


if __name__ == "__main__":
    main()
    # # Initialize the model, optimizer and loss function
    # model = Net()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.MSELoss()

    # # Load data
    # train_loader, test_loader = load_data()

    # # Train the model
    # train(model, train_loader, optimizer, loss_fn, epochs=5)

    # # Evaluate the model
    # evaluate(model, test_loader)