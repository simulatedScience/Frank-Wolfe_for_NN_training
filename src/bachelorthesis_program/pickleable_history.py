"""
This module provides functions for saving the training history of a neural network.

Author: Sebastian Jost
"""

def get_save_history(history_dict, model_params, optimizer_params, model_path, train_time, save_time):
    """
    convert a given tensorflow history object into a custom pickleable object, which saves the important information.
    """
    return pickleable_history(
            history_dict,
            model_params=model_params, 
            optimizer_params=optimizer_params,
            save_time=save_time,
            model_path=model_path,
            train_time=train_time)

class pickleable_history():
    def __init__(self, history, model_params, optimizer_params,
            save_time=None, test_scores=None, model_path=None, train_time=None):
        # save info about experiment setup
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        # save info about training
        self.history = history
        if save_time is not None:
            self.save_time = save_time
        if train_time is not None:
            self.train_time = train_time
        self.final_loss = history["loss"][-1]
        if "val_loss" in history.keys():
            self.final_val_loss = history["val_loss"][-1]
        # save info about model file
        if model_path is not None:
            self.model_path = model_path
        # add possibility of specifying test scores directly
        if test_scores is not None:
            self.test_scores = test_scores

    def add_test_scores(self, test_scores):
        """
        add given test scores from `model.evaluate(...)` to the history object
        """
        self.test_scores = test_scores

    def summary(self):
        """
        print a summary about the model used as well as the training parameters and results.
        """
        # TODO
        raise NotImplementedError