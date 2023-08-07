"""
only change a single parameter in many different settings of the other parameters.
This yields a much more reliable overview of the effect of the parameter as well as it's correlation to other parameters.

Author: Sebastian Jost
"""
import os
import glob

import numpy as np

from helper_functions import build_filepath, dict_cross_product
from batch_summary import summarize_batch
from load_md_files import load_study_info
from top_n_list import top_n_list
from file_management import save_picklable_object, load_pickle_file, save_parameter_analysis




def statistical_analysis_of_study(study_folder):
    """
    calculate statistical information about each batch of a parameter study (mean, standard deviation, min and max for several metrics)
    these findings will be saved in the batch folders to be used for further analysis without re-calculating these values

    inputs:
    -------
        study_folder - (str) - name of the study folder

    returns:
    --------
        (str) - absolute filepath to a dictionary mapping parameter settings to the filepath of the corresponding performance summary
    """
    # initialize a dictionary mapping parameter settings to the filepath of the corresponding performance summary
    summary_files = dict()
    filename = "*batch"
    filepath = build_filepath(filename, study_folder, ending="")
    # loop through all batch folders through pattern matching with the above filepath
    for batch_folder in glob.glob(filepath):
        # only use batch folder name from filepath
        batch_folder = os.path.split(batch_folder)[-1]
        # summarize model performance
        summary_filepath, param_dict, optimizer_keys = summarize_batch(
            study_folder, batch_folder)
        param_dict["optimizer_params"] = {key: param_dict[key] for key in optimizer_keys}
        for key in optimizer_keys:
            del param_dict[key]
        hashable_params = get_hashable_params(param_dict)
        summary_files[hashable_params] = summary_filepath
    # save dictionary mapping parameters to summary filepaths
    summary_filepath = save_picklable_object(
        summary_files, filename="summary_filepath_dict", sub_folders=study_folder)
    return summary_filepath


def analyse_parameter_study(study_folder, summary_filepaths_filename, stat_metric="avg"):
    """
    Analyse a parameter study given by a folder path
    That `study_folder` should contain folders for trained batches.
    each batch contains one or more models trained with the same parameters but potentially different random starting values.
    models from different batches have different parameter settings.

    The Analysis will try to find the best value for each parameter as well as the overall best and worst combinations of all parameters in several different categories.
    To compare parameter values, loop through all values of the other parameters and for each case determine which parameter value yields the better results.
    The count how often one parameter value delivers better results than other values and the differences in the results are saved and used to determine which value is better.

    inputs:
    -------
        study_folder - (str) - name of the study folder
        summary_filepaths_filename - (str) - filepath or -name of the dictionary mapping parameter settings to the corresponding summary files
        stat_metric - (str) - statistical value representing batch performance in each category. May be ('avg', 'std', 'min' or 'max')

    returns:
    --------
        (str) - absolute filepath to the file containing the results of the parameter analysis
    """
    study_settings, optimizer_params_list = load_study_info(
        filename="study_parameters", study_folder=study_folder)
    summary_filepath_dict = load_pickle_file(
        summary_filepaths_filename, study_folder)

    min_max_lists_exist = False
    # for each parameter, for each metric, save a score for every explored value, counting the number of times it was the best value
    param_count_results = dict()
    # for each parameter, for each metric, save a score for every explored value, measuring the average loss relative to the best explored value
    param_diff_results = dict()
    # for each parameter, for each metric, save the best value of the parameter
    param_best_values = dict()
    # loop through all parameter names and their value lists
    for current_param, param_values in study_settings.items():
        # check whether or not there are mutliple values for the current parameter in the study
        if not isinstance(param_values, list):
            continue
        # prepare dict for loops
        comparison_dict = study_settings.copy()
        del comparison_dict[current_param]
        # initialize lists for comparing parameter values except in the first iteration
        if min_max_lists_exist:
            param_value_best_count = {
                key: [0]*len(param_values) for key in min_max_lists.keys()}
            param_value_differences = list()
        # loop through all combinations of the other parameters
        product_generator, key_list = dict_cross_product(comparison_dict)
        for param_list in product_generator:
            param_dict = {key: value for key,
                          value in zip(key_list, param_list)}
            summary_dicts = [0]*len(param_values)
            # param_hashes = [""]*len(param_values)
            param_dicts = [0]*len(param_values)
            # loop through all possible values for the current parameter
            for i, param_value in enumerate(param_values):
                # add current parameter value to dict for hashing
                param_dict[current_param] = param_value
                hashable_params = get_hashable_params(param_dict)
                # param_hashes[i] = hashable_params
                param_dicts[i] = dict(param_dict)  # copy dict
                # load summary file corresponding to current parameter settings
                sumary_filepath = summary_filepath_dict[hashable_params]
                training_summary_dict = load_pickle_file(sumary_filepath)
                for key in list(training_summary_dict.keys()):
                    training_summary_dict[f"{key}_{stat_metric}"] = training_summary_dict[key]
                    del training_summary_dict[key]
                # save summary dictionaries for all choices of the current parameter
                summary_dicts[i] = training_summary_dict
            # initialize lists for tracking total best and worst parameter settings
            if not min_max_lists_exist:
                min_max_lists = init_min_max_lists(training_summary_dict)
                min_max_lists_exist = True
                # initialize lists for comparing parameter values
                param_value_best_count = {
                    key: [0]*len(param_values) for key in min_max_lists.keys()}
                param_value_differences = list()
            # actually analyse the data now
            winner_indeces, diffs_lists = analyse_parameter_values(
                # param_values,
                summary_dicts,
                min_max_lists,
                param_dicts,
                # param_hashes,
                n_repetitions=param_dict["number_of_repetitions"],
                stat_metric=stat_metric,
                relative_diff=True)
            # add 1 to win count for the best parameter setting in each category
            for key, winner_index in winner_indeces.items():
                param_value_best_count[key][winner_index] += 1
            param_value_differences.append(diffs_lists)
        # save results of parameter comparison
        param_count_results[current_param] = param_value_best_count
        param_diff_results[current_param] = get_avg_diffs(
            param_value_differences, len(param_values))
        param_best_values[current_param] = get_winners(
            param_values,
            param_value_best_count,
            param_diff_results[current_param])

    # calculate win ratios from win counts
    param_win_ratios = calc_param_win_ratios(param_count_results)
    # save all results to one file
    results_filepath = save_parameter_analysis(
        min_max_lists,
        param_count_results,
        param_win_ratios,
        param_diff_results,
        param_best_values,
        study_folder,
        filename=f"parameter_analysis_{stat_metric}",)
    return results_filepath


def calc_param_win_ratios(param_count_dict):
    """


    inputs:
    -------
        param_count_results - (dict) - dictionary containing win counts all parameters and for all categories
            keys - (str) - parameter names
            values - (dict) - dictionary containing results for all categories
                keys - (str) - matric name
                values - (list) - win count for each parmaeter setting. Within one parameter name, these lists have the same length
    """
    win_ratio_dict = dict()
    # loop through all parameter names
    for param_name, category_dict in param_count_dict.items():
        win_ratio_dict[param_name] = dict()
        # compute total number of wins possible
        any_sublist = next(iter(category_dict.values()))
        total = sum(any_sublist)
        # loop through all categories
        for category_name, win_count_list in category_dict.items():
            # calculate win ratios from win counts
            win_ratio_list = [win_count/total for win_count in win_count_list]
            win_ratio_dict[param_name][category_name] = win_ratio_list
    return win_ratio_dict


def get_winners(param_values, param_value_best_count, diff_results):
    """
    Automatically determine which parameter choice is the best for each metric
    A parameter choice is a 'winner', if it has the highest win count AND the lowest average difference from the best value.
    If no parameter fulfills this condition, 'winner' is 'undetermined'.
    If multiple parameters have the same values, the 'winner' is given as one of the parameter choices.

    inputs:
    -------
        param_values - (list) - list of all explored values for the currently examined parameter
        param_value_best_count - (dict) - keys are metrics used to compare performance
            values are lists of integers representing how often each parameter value yielded the best results when keeping all other parameters constant
        diff_results - (dict) - keys are metrics used to compare performance
            values are lists of floats representing the average difference to the result of the best performing parameter value when keeping all other parameters constant

    returns:
    --------
        (dict) - for each metric, contains the best parameter value.
            if that is not clear, the value will be 'undetermined'
    """
    winner_dict = dict()
    for metric_key in param_value_best_count.keys():
        # if "mae" in metric_key:
        #     continue
        count_has_different_values = len(
            set(param_value_best_count[metric_key])) > 1
        diff_has_different_values = len(set(diff_results[metric_key])) > 1
        bool_pair = (count_has_different_values, diff_has_different_values)
        if bool_pair == (False, False):
            winner_dict[metric_key] = "undetermined"
        elif bool_pair == (True, False):
            index = np.argmax(param_value_best_count[metric_key])
            winner_dict[metric_key] = param_values[index]
        elif bool_pair == (False, True):
            index = np.argmin(diff_results[metric_key])
            winner_dict[metric_key] = param_values[index]
        else:
            best_index_count = np.argmax(param_value_best_count[metric_key])
            best_index_diff = np.argmin(diff_results[metric_key])
            if best_index_count == best_index_diff:
                winner_dict[metric_key] = param_values[best_index_count]
            else:
                winner_dict[metric_key] = "undetermined"
    return winner_dict


def get_avg_diffs(param_value_differences, n_param_values):
    """
    Calculate the average differences or relative differences between result values of each parameter 
    and the best result value.
    
    Inputs:
    -------
    param_value_differences - (list[dict])
        Dictionaries containing difference lists for each category.
        Each dictionary represents one choice of all other parameters.
    n_param_values - (int)
        Number of explored values of the currently examined parameter.

    Returns:
    --------
    diff_sums : dict
        Dictionary containing the average differences or relative differences for each parameter value.
    """
    counter = 0
    diff_sums = {key: np.zeros(n_param_values) for key in param_value_differences[0].keys()}
    for diff_dict in param_value_differences:
        counter += 1
        for key, diff_array in diff_dict.items():
            diff_sums[key] += diff_array
    # divide by number of parameter settings
    for key in diff_sums.keys():
        diff_sums[key] /= counter
    return diff_sums


def analyse_parameter_values(
        summary_dicts,
        min_max_lists,
        param_dicts,
        n_repetitions=1,
        stat_metric="avg",
        relative_diff=False):
    # param_values,
    """
    Compare the given parameter values for a single parameter while keeping all other parameters constant.
    The different values are evaluated based on results given by `summary_dicts`, summarizing the performance of each model.
    Some results of comparison are saved in `min_max_lists`, others are returned.

    inputs:
    -------
    # param_values - (list) - list of settings for one changing parameter
    summary_dicts - (list) of (dict) - dicts summarizing performance with respect to each metric for all given parameter values
    min_max_lists - (list) of (dict) of (top_n_list) - minimum and maximum top_n_lists for each metric being analyzed
    param_dicts - (list) of (dict) - parameter setting dictionaries
        These will be used as keys in the min_max_lists.
    stat_metric - (str) - string specifying which statistical value is used for analysis. Possible values are:
        'min' - minimal final value
        'max' - maximal final value
        'avg' - average final value
        'std' - standard deviation of final value

    returns:
    --------
    winner_indeces - (dict) - dictionary containing indeces of the best parameter in param_values for each category (given by the key)
    diff_lists - (dict) - dictionary containin differences to the best value for each category.
    """
    winner_indeces = dict()
    diff_lists = dict()
    ignore_keys = list(min_max_lists.keys())
    
    for metric, (min_list, max_list) in min_max_lists.items():
        
        if "accuracy" in metric:
            value_sign = -1
        else:
            value_sign = 1
        
        result_values = np.zeros(len(summary_dicts))
        best_value = value_sign * np.inf
        
        for i, (summary_dict, param_dict) in enumerate(zip(summary_dicts, param_dicts)):
            
            for temp_metric in min_max_lists.keys():
              if n_repetitions > 1:
                for temp_stat_metric in ["min", "max", "avg", "std"]:
                  param_dict[f"{temp_metric[:-4]}_{temp_stat_metric}"] = summary_dict[temp_metric][temp_stat_metric]
              else:
                param_dict[f"{temp_metric[:-4]}_{stat_metric}"] = summary_dict[temp_metric][stat_metric]
            
            mae_loss_metric = metric[:-4] + "_mae_" + stat_metric
            if "loss" in metric and mae_loss_metric in min_max_lists.keys():
                result_values[i] = summary_dict[mae_loss_metric][stat_metric]
                ignore_keys.append(mae_loss_metric)
            else:
                result_values[i] = summary_dict[metric][stat_metric]

            min_list.add_item(param_dict, result_values[i], ignore_keys)
            max_list.add_item(param_dict, result_values[i], ignore_keys)

            if value_sign*result_values[i] < value_sign*best_value:
                best_index = i
                best_value = result_values[i]
        
        winner_indeces[metric] = best_index
        
        if relative_diff:
            # calculate the relative differences instead of absolute differences
            diff_lists[metric] = 100 * (value_sign * (result_values - best_value) / best_value)
        else:
            diff_lists[metric] = value_sign * (result_values - best_value)
    
    return winner_indeces, diff_lists



def init_min_max_lists(summary_dict, n_elems=5):
    """
    initialize `top_n_list` objects for tracking the best and worst parameter settings for each metric

    inputs:
    -------
        summary_dict - (dict) - for each key in the summary dict, one min- and one max list will be created.
        n_elems - (int) - number of elements saved in each list

    returns:
    --------
        (dict) - dictionary with the same keys as `summary_dict` and pairs of min- and max lists as values.
            these lists will be used to store the overall best and worst perameter choices and their values for each metric
    """
    min_max_lists = dict()
    for key in summary_dict:
        best_n_list = top_n_list(n_elems, mode="max")
        worst_n_list = top_n_list(n_elems, mode="min")
        min_max_lists[key] = (worst_n_list, best_n_list)
    return min_max_lists


def get_hashable_params(param_dict):
    """
    convert a parameter dictionary into a hashable string used for quickly finding training files for a given parameter setting

    inputs:
    -------
        param_dict - (dict) - dictionary specifying all parameters for training the neural network

    returns:
    --------
        (str) - string of the parameter choices
    """
    # a hardcoded order of dict keys
    ordered_keys = [
        "neurons_per_layer",
        "input_shape",
        "output_shape",
        "activation_functions",
        "last_activation_function",
        "layer_types",
        "loss_function",
        "training_data_percentage",
        "number_of_epochs",
        "batch_size",
        "validation_split",
        "number_of_repetitions",
    ] + list(param_dict["optimizer_params"].keys())
    ordered_keys.sort()
    copied_dict = param_dict.copy()
    for key in param_dict["optimizer_params"].keys():
        copied_dict[key] = param_dict["optimizer_params"][key]
    
    parameter_tuple = tuple(copied_dict[key] for key in ordered_keys)
    return str(parameter_tuple)


if __name__ == "__main__":
    import time
    import tkinter as tk
    from tkinter import filedialog
    # study_folder = tk.filedialog.askdirectory()
    # study_folder = "2021-09-09_21-56-12_parameter_study_debug"
    # study_folder = "bachelor_thesis_parameter_study_mnist"
    # study_folder = "bachelor_thesis_pstudy_mnist_2"
    # study_folder = "bachelor_thesis_adam_variants"
    # study_folder = "2022-05-24_21-55-21_parameter_study"
    # study_folder = "bachelor_thesis_parameter_study_chem"
    # study_folder = "bachelor_thesis_param_study_fastcAdam"
    # study_folder = "2021-10-17_00-01-14_parameter_study"
    # study_folder = "C:\\future_D\\uni\\Humboldt Uni\\Nebenhoerer SoSe 2023\\FW_NN_training\\src\\bachelorthesis_program\\training_info\\2023-08-06_20-41-19_parameter_study"
    # study_folder = "C:\\future_D\\uni\\Humboldt Uni\\Nebenhoerer SoSe 2023\\FW_NN_training\\src\\bachelorthesis_program\\training_info\\2023-08-07_00-24-41_mnist_sfw_parameter_study"
    # study_folder = "C:\\future_D\\uni\\Humboldt Uni\\Nebenhoerer SoSe 2023\\FW_NN_training\\src\\bachelorthesis_program\\training_info\\2023-08-07_10-43-21_mnist_reg_sfw_parameter_study"
    study_folder = "C:\\future_D\\uni\\Humboldt Uni\\Nebenhoerer SoSe 2023\\FW_NN_training\\src\\bachelorthesis_program\\training_info\\2023-08-07_15-30-46_chemreg_reg_sfw_parameter_study"
    start_time = time.time()
    summary_filepaths_filepath = statistical_analysis_of_study(study_folder)
    end_time = time.time()
    print(f"statistical summaries took {end_time-start_time:.3f}s")

    # summary file mnist old
    # summary_filepaths_filepath = r"d:\Uni\Humboldt Uni\Semester 4\Bachelorarbeit\programme\neural net testing program v2\training_info\bachelor_thesis_parameter_study_chem\summary_filepath_dict.pickle"
    # summary file mnist new (with correct cAdam)
    # summary_filepaths_filepath = 'summary_filepath_dict.pickle'
    # summary_filepaths_filepath = r"d:\Uni\Humboldt Uni\Semester 4\Bachelorarbeit\programme\neural net testing program v2\training_info\bachelor_thesis_pstudy_mnist_2\summary_filepath_dict.pickle"
    print(f"summary_filepaths_filepath = \n{summary_filepaths_filepath}")

    # start parameter analysis with time tracking
    start_time = time.time()
    analyse_parameter_study(study_folder, summary_filepaths_filepath, stat_metric="avg")
    end_time = time.time()
    print(f"analysis took {end_time-start_time:.3f}s")
