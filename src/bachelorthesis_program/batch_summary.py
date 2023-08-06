"""
this module implements functions for summarizing the performance of one batch of trained models.

Author: Sebastian Jost
"""
import glob
from os import stat # file searching

import numpy as np

from file_management import load_pickle_file, save_picklable_object
from helper_functions import build_filepath
from load_md_files import load_batch_info
from loss_conversion import loss_conversion


def summarize_batch(study_folder, batch_folder):
    """
    create a statistical analysis of a batch of trained models.

    inputs:
    -------
        study_folder - (str) - name of the study folder
        batch_folder - (str) - name of the batch folder

    returns:
    --------
        stat_filename - (str) - absolute filepath for the file where the results of the statistical analysis are saved as `statistical_results.pickle` details about the saved data structure can be found in `get_result_lists`
        params - (dict) - dictionary containing the parameter names and values used for training the analyzed batch
        optimizer_keys - (list) of (str) - list of keys passed to the optimizer. These can be completely customized.
            The corresponding values are saved in the parameter dictionary `params` that also gets returned.
    """
    # load parameters used for training the given batch
    params, optimizer_keys = load_batch_info("batch_parameters", study_folder, batch_folder)
    if params["output_shape"] in (1, (1,)): # save loss function used to later convert loss to MAE
        loss_func = params["loss_function"]
    else:
        loss_func = None
    repetitions = params["number_of_repetitions"]
    # collect results from all repetitions of training the model with the loaded parameters
    result_info = get_result_lists(study_folder, batch_folder, repetitions)
    # calculate statistical information summarizing the performance of the trained models
    stat_results = batch_statistical_analysis(result_info, loss_func)
    # save results
    stat_filename = save_picklable_object(stat_results, filename="statistical_results", sub_folders=[study_folder, batch_folder])
    return stat_filename, params, optimizer_keys


def get_result_lists(study_folder, batch_folder, repetitions=1):
    """
    load training history from all files in the given batch folder and save all relevant information into one dictionary containing lists of all final values achieved by each model

    inputs:
    -------
        study_folder - (str) - name of the study folder
        batch_folder - (str) - name of the batch folder
        repetitions - (int) - number of models trained per batch (-> each model within a batch has identical parameters except for random starting values and randomness in the validation split and training order)

    returns:
    --------
        result_info - (dict) - dictionary containing keys:
            'training_time' - measured time for training the model. (does not include model creation or saving times)
            'final_loss' - final loss during training
            'final_val_loss' - final validation loss during training
            'test_loss' - loss during testing
            if accuracy is tracked, also contains:
            'final_accuracy' - final accuracy during training
            'final_val_accuracy' - final validation accuracy during training
            'test_accuracy' - accuracy during testing
            for each key the value is a numpy array with the corresponding value for each model that was trained with the same parameters (= within the same batch)
    """
    # build a filepath pattern to all relevant training_history files
    filepath = build_filepath("*training_history.pickle", study_folder, batch_folder, ending="")
    result_info = dict()
    # prepare lists for the training values as appending to lists is slow
    result_info["training_time"] = np.zeros(repetitions)
    # loss
    result_info["final_loss"] = np.zeros(repetitions)
    result_info["final_val_loss"] = np.zeros(repetitions)
    result_info["test_loss"] = np.zeros(repetitions)
    # loop through all training history files through pattern matching the filepath defined above
    for i, filename in enumerate(glob.glob(filepath)):
        history_obj = load_pickle_file(filename, study_folder, batch_folder)
        # add metrics to lists for statistical analysis
        result_info["training_time"][i] = history_obj.train_time
        result_info["final_loss"][i] = history_obj.final_loss
        result_info["final_val_loss"][i] = history_obj.final_val_loss
        if "accuracy" in history_obj.history.keys():
            # initialize accuracy lists if necessary
            if i == 0:
                result_info["final_accuracy"] = np.zeros(repetitions)
                result_info["final_val_accuracy"] = np.zeros(repetitions)
                result_info["test_accuracy"] = np.zeros(repetitions)
            # save accuracy values and test results for the current model
            result_info["final_accuracy"][i] = history_obj.history["accuracy"][-1]
            result_info["final_val_accuracy"][i] = history_obj.history["val_accuracy"][-1]
            result_info["test_accuracy"][i] = history_obj.test_scores[1]
            result_info["test_loss"][i] = history_obj.test_scores[0]
        else: # only one metric tracked during evaluation (loss)
            result_info["test_loss"][i] = history_obj.test_scores
    return result_info


def batch_statistical_analysis(result_info, loss_func=None):
    """
    calculate min, max, average and standard deviation for all lists in the `result_info` dictionary.

    inputs:
    -------
        result_info - (dict) - keys are all tracked metrics, values are arrays with the corresponding values achieved during the repetitions of training models with the same parameters
        see `get_result_lists` for more details
        loss_func - (str) - sting specifying the loss function used (if applicable). If this is specified, an additional entry 'mae_loss' gets added to the returned dict, that contains statistical information about mean squared error loss.
            note, that this conversion only works for single-node outputs of models.

    returns:
    --------
        (dict) - keys are the same names of metrics as in the input. The array values get replaced with dictionaries containing the mean, standard deviation, minimum and maximum of each array.
    """
    stat_results = dict()
    for key, value_list in result_info.items():
        if loss_func is not None and "loss" in key:
            mae_loss_values = loss_conversion(value_list, loss_func)
            stat_results[key+"_mae"] = statistical_analysis(mae_loss_values)
        stat_results[key] = statistical_analysis(value_list)
    return stat_results


def statistical_analysis(value_list):
    """
    calculate min, max, average and standard deviation for the given list of numerical values.
    results are saved in a dictionary with keys `min`, `max`, `avg` and `std`

    inputs:
    -------
        value_list - (iterable) - any list or array of numerical values
            tested for 1d numpy arrays

    returns:
    --------
        (dict) - dictionary containing the mean, standard deviation, minimum and maximum of the given values
    """
    stat_dict = dict()
    stat_dict["min"] = np.min(value_list)  # minimum
    stat_dict["max"] = np.max(value_list)  # maximum
    stat_dict["avg"] = np.mean(value_list) # average
    stat_dict["std"] = np.std(value_list)  # standard deviation
    return stat_dict


if __name__ == "__main__":
    # print(summarize_batch("bachelor_thesis_parameter_study_mnist", "2021-09-10_15-20-25_batch"))
    print(summarize_batch("bachelor_thesis_parameter_study_mnist", "2021-09-09_22-09-06_batch"))
    # print(summarize_batch("bachelor_thesis_parameter_study_chem", "2021-09-10_00-08-26_batch"))