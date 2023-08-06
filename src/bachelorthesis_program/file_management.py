"""
this module includes all file saving and loading functions used for this program.

Author: Sebastian Jost
"""
import os
import pickle

import torch

from helper_functions import get_date_time, get_max_key_length, build_filepath

#############################################
# file management functions before training #
#############################################

def create_study_folder(network_params, training_params, optimizer_params, study_folder=None):
    """
    create a new folder for the current parameter study.

    in that folder a human-readable file containing all relevant information will be saved.

    returns:
    --------
        (str) - study folder name
    """
    try: # try creating a `training_info` folder
        os.mkdir(os.path.join(os.path.dirname(__file__), "training_info"))
        print(f"created new directory `training_info`")
    except FileExistsError:
        pass
    if study_folder is None:
        date_time = get_date_time()
        study_folder = date_time + "_parameter_study"
    try: # try creating a folder for the parameter study
        os.mkdir(os.path.join(os.path.dirname(__file__), "training_info", study_folder))
        print(f"created new directory at {os.path.join('training_info', study_folder),}")
    except FileExistsError:
        pass
    save_study_params(network_params, training_params, optimizer_params, study_folder)
    return study_folder


def save_study_params(network_params, training_params, optimizer_params_list,
        study_folder, filename="study_parameters.md"):
    """
    save all parameters of the parameter study as a human-readable file.
    """
    filepath = os.path.join(os.path.dirname(__file__), "training_info", study_folder, filename)
    padding_len = 2 + get_max_key_length(network_params, training_params, *optimizer_params_list)
    with open(filepath, "w") as file:
        file.write("## all parameters used in this study: \n\n")
        file.write("\n### Neural Network parameters:\n")
        for key, value in network_params.items():
            file.write(f"   - {key+':':{padding_len}} {value}\n")
        file.write("\n### Training parameters:\n")
        for key, value in training_params.items():
            file.write(f"   - {key+':':{padding_len}} {value}\n")
        file.write("\n### Optimizer setups:\n")
        for i, optimizer_settings in enumerate(optimizer_params_list):
            file.write(f"{i}. optimizer setup\n")
            for key, value in optimizer_settings.items():
                file.write(f"   - {key+':':{padding_len}} {value}\n")


def create_batch_folder(model_params, optimizer_params, study_folder=None, batch_folder=None):
    """
    create a folder for one batch of model training. All models within this folder will use the same parameters for training.

    returns:
    --------
        (str) - batch folder name
    """
    if batch_folder is None:
        date_time = get_date_time()
        batch_folder = date_time + "_batch"
    try:
        os.mkdir(os.path.join(os.path.dirname(__file__), "training_info", study_folder, batch_folder))
        print(f"created new directory at {os.path.join('training_info', study_folder, batch_folder)}")
    except FileExistsError:
        pass
    save_batch_info(model_params, optimizer_params, study_folder, batch_folder)
    return batch_folder


def save_batch_info(model_params, optimizer_params,
        study_folder, batch_folder, filename="batch_parameters.md"):
    """
    save all parameters used to train the batch in the given folder as a human-readable file.
    """
    filepath = os.path.join(os.path.dirname(__file__), "training_info", study_folder, batch_folder, filename)
    padding_len = 2 + get_max_key_length(model_params, optimizer_params)
    with open(filepath, "w") as file:
        file.write("## all parameters used in this batch of the study: \n\n")
        file.write("\n### Model and training parameters:\n")
        for key, value in model_params.items():
            file.write(f"   - {key+':':{padding_len}} {value}\n")
        file.write("\n### Optimizer parameters:\n")
        for key, value in optimizer_params.items():
            file.write(f"   - {key+':':{padding_len}} {value}\n")


############################################
# file management functions after training #
############################################

def save_model(model, filename="neural_network.pt", save_time=None, sub_folders=None):
    """
    save tensorflow model to the folder `training_info` with the given filename.
    if `save_time` is given, it gets added to the beginning of the filename

    inputs:
    -------
        history - any pickleable type - any picklable object. Usually a customized history object returned from tensorflow model.fit
        filename - (str) - name for the saved model folder
        save_time - (str) - a string specifying date and time that gets added to the beginning of the filename
        sub_folders - (str) or (list) of (str) - 

    returns:
    --------
        None
    """
    try: # create a "training_info" folder if it doesn't exist yet
        os.mkdir(os.path.join(os.path.dirname(__file__), "training_info"))
        print(f"created new directory `training_info`")
    except FileExistsError:
        pass
    if save_time is not None:
        filename = save_time + "_" + filename
    if sub_folders is None:
        filepath = os.path.join(os.path.dirname(__file__), "training_info", filename)
    else:
        if isinstance(sub_folders, str):
            sub_folders = (sub_folders,)
        filepath = os.path.join(os.path.dirname(__file__), "training_info", *sub_folders, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # filepath = "C:\\future_D\\uni\\Humboldt Uni\\Nebenhörer SoSe 2023\\test_model.pt"
    # with open(filepath, "w") as file:
    #     file.write("working")
    torch.save(model, filepath)
    # sub_path = os.path.join("training_info", *sub_folders, filename)
    # print(f"saved model to {sub_path}")
    return filepath


def save_picklable_object(history, filename="training_history", save_time=None, sub_folders=None, ending=".pickle"):
    """
    save history object containing information about the training process of a neural network to `training_info` folder
    if `save_time` is given, it gets added to the beginning of the filename

    inputs:
    -------
        history - any pickleable type - any picklable object. Here usually a History object returned from tensorflow model.fit
        filename - (str) - file name for the saved file (without file extension `.pickle`)
        save_time - (str) - a string specifying date and time that gets added to the beginning of the filename

    returns:
    --------
        None
    """
    try: # create a "training_info" folder if it doesn't exist yet
        os.mkdir(os.path.join(os.path.dirname(__file__), "training_info"))
        print(f"created new directory `training_info`")
    except FileExistsError:
        pass
    if save_time is not None:
        filename = save_time + "_" + filename
    if not ending in filename:
        filename += ending
    if sub_folders is None:
        filepath = os.path.join(os.path.dirname(__file__), "training_info", filename)
    else:
        if isinstance(sub_folders, str):
            sub_folders = (sub_folders,)
        filepath = os.path.join(os.path.dirname(__file__), "training_info", *sub_folders, filename)
    with open(filepath, "wb") as file:
        pickle.dump(history, file, protocol=4)
    # sub_path = os.path.join("training_info", *sub_folders, filename)
    # print(f"saved training history to {sub_path}")
    return filepath

#####################################
# file management for data analysis #
#####################################

def load_pickle_file(filename, study_folder=None, batch_folder=None):
    """
    load any pickleable object from the given filename
    """
    filepath = build_filepath(filename, study_folder, batch_folder, ending=".pickle")
    try:
        with open(filepath, "rb") as file:
            history_obj = pickle.load(file)
        return history_obj
    except FileNotFoundError:
        print(f"File not found at {filepath}")


def save_parameter_analysis(
        min_max_lists,
        param_count_results,
        param_win_ratios,
        param_diff_results,
        param_best_values,
        study_folder,
        filename="parameter_analysis_results"):
    """
    save results of the analysis of a parameter study to a file from where they can be visualized
    """
    parameter_analysis_results = {
            "min_max_lists":min_max_lists,
            "param_count_results":param_count_results,
            "param_win_ratios":param_win_ratios,
            "param_diff_results":param_diff_results,
            "param_best_values":param_best_values}
    return save_picklable_object(parameter_analysis_results, filename=filename, sub_folders=[study_folder])


def load_parameter_analysis(filename="parameter_analysis_results", study_folder=None):
    """
    load the results of a parameter analysis

    filename may also be a complete path to the file (including file ending)
    otherwise study_folder must be specified
    
    returns:
    --------
        (dict) - min_max_lists        for each parameter and each category
        (dict) - param_count_results  for each parameter and each category
        (dict) - param_win_ratios     for each parameter and each category
        (dict) - param_diff_results   for each parameter and each category
        (dict) - param_best_values    for each parameter and each category
    """
    parameter_analysis_results = load_pickle_file(filename, study_folder=study_folder)
    min_max_lists = parameter_analysis_results["min_max_lists"]
    param_count_results = parameter_analysis_results["param_count_results"]
    param_win_ratios = parameter_analysis_results["param_win_ratios"]
    param_diff_results = parameter_analysis_results["param_diff_results"]
    param_best_values = parameter_analysis_results["param_best_values"]
    return min_max_lists, param_count_results, param_win_ratios, param_diff_results, param_best_values