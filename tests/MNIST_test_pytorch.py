# Import necessary modules
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define the model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # define layers of the NN
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(28*28, 128)
        self.output = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model: How an input `x` is processed to make a prediction.

        Args:
            x (torch.Tensor): Input tensor to the model.

        Returns:
            torch.Tensor: Output of the model.
        """
        # define how the model is used to make predictions
        # MNIST uses 28x28 images, so we need to flatten them to 784x1
        x = self.flatten(x)
        x = torch.relu(self.hidden(x))
        x = nn.functional.log_softmax(self.output(x), dim=1)
        return x

def load_data(batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    """
    Load the MNIST dataset. If it does not exist, it will be downloaded.

    Args:
        batch_size (int, optional): Batch size for the data loaders. Defaults to 64.

    Returns:
        (DataLoader): Training data loader.
        (DataLoader): Testing data loader.
    """
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.MNIST(root='./datasets', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./datasets', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader

def train(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        epochs: int = 5):
    # Train the model
    for epoch in range(epochs): 
        # create progress bar to show progress of epoch
        loop = tqdm(train_loader, total=len(train_loader), leave=True, ncols=100)
        for images, labels in loop:
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            # update progress bar
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(loss = f"{loss.item():.3f}")

def evaluate(model: nn.Module, test_loader: DataLoader):
    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy: ', 100 * correct / total)

if __name__ == "__main__":
    # Initialize the model, optimizer and loss function
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Load data
    train_loader, test_loader = load_data()

    # Train the model
    train(model, train_loader, optimizer, loss_fn, epochs=5)

    # Evaluate the model
    evaluate(model, test_loader)