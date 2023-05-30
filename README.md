# Deep Neural Network Training with Frank–Wolfe Algorithm
This project aims to replicate some of the experiments from the paper (Deep Neural Network Training with Frank–Wolfe)[https://arxiv.org/abs/2010.07243].

Expanding on the paper, we will directly compare the algorithm to the popular Adam optimizer and apply it to a different dataset.

## Requirements
- Python 3.9
- PyTorch 1.11.0 -> as framework for neural networks
- PyTorch lightning 1.6 -> to simplify some pytorch code

## TOODs
- Implementation:
  - [ ] Implement Frank-Wolfe algorithm
  - [ ] Implement different contraints
  - [ ] Track metrics during training
  - [ ] Implement visualizations
- Experiments:
  - [ ] Test SFW on different datasets, comparing to SGD & Adam
    - [ ] MNIST
    - [ ] Fashion-MNIST
    - [ ] ChemReg (first layer with sparse constraints, after that L2 regularization/ constraints)
  - [ ] create diagrams to visualize the results
  - [ ] explain the results
- Presentation:
  - [ ] create slides to explain the algorithm & constraints
  - [ ] create slides to explain the experiments
  - [ ] create slides to explain the results