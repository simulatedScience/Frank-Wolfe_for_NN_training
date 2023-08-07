"""
this module implements functions for evaluating performance of trained neural networks

Author: Sebastian Jost
"""

import torch

# from helper_functions import auto_use_multiprocessing

def test_model(model, test_data, batch_size=1000) -> float | list[float]:
    """
    evaluate the performance of a given model based on given data.
    
    Args:
        model (keras.Model): the model to be evaluated
        test_data (tuple): (x_test, y_test)
        batch_size (int, optional): batch size for evaluation. Defaults to 1000.

    Returns:
        float | list[float]: loss value or list of loss value and accuracy
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x_test, y_test = test_data
    model.init_testing()
    loss_value, test_outputs = model.evaluate(
            x_test,
            y_test,
            batch_size=batch_size,
            device=device,
            test_mode=True)
    if model.track_accuracy:
        accuracy = model.history['test_accuracy'][-1]
        test_scores = [loss_value, accuracy]
    else:
        test_scores = loss_value
    return test_scores
