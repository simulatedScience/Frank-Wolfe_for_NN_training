This file summarizes the file structure for this program.

### Workflow for experiments
Decisions to make:
- dataset
- preprocessing for data
- neural network architecture (number of layers, neurons, types of neurons, etc.)
- optimizer to use (SGD/ Adam/ SFW)
  - optimizer parameters:
    - learning rate,
    - momentum parameters (if applicable)
    - constraints (for SFW)
- training time (number of epochs, maximum time)
- metrics to track (loss, MAE loss, accuracy, f1-score, etc.)