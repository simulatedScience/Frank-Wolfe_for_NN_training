"""
This module provides functions for printing the best and worst performing models' parameters in a table.
Each network's parameters are printed in a separate column, while the rows represent the different hyperparameters.
These tables are saved to a .tex file, formatted as Latex code.

The hyperparameters shown in the table can be adjusted in `get_default_param_list`.
To improve text output, there is a function `get_param_setting_renaming` that replaces certain texts
  to make them suitable for Latex or abbreviate them to better fit in the tables.

Author: Sebastian Jost
"""

import sys
import os

import numpy as np
# from loss_conversion import loss_conversion


def print_best_worst_vertical(
        min_max_lists,
        metrics,
        prec: int = 10,
        param_setting_renaming: str = "default",
        color: str = "equalParamColor",
        study_folder: str = None,
        label_prefix: str = "",
        expriment_name: str = None):
    """
    inputs:
    -------
        min_max_lists - (dict) of (str):(pair) of (top_n_list) - the dictionary contains min- and max-lists for each metric that was tracked during training. those are saved as pairs (min_list, max_list) as the values of that dictionary.
        metrics - (list) of (str) - list of keys of `min_max_lists` only results for those metrics will be printed.
        prec - (int) - number of decimal places to which to round the result scores for each given metric.
        param_setting_renaming - (dict) of (str):(str) or "default" - if any parameter name or setting is in the keys of this dictionary, it gets replaced with the corresponding value.
            "default" sets this dictionary using `get_param_setting_renaming()`.
    """
    if param_setting_renaming == "default":
        param_setting_renaming = get_param_setting_renaming()
    # determine if each model was trained more than once
    if min_max_lists[metrics[0]][0].keys[0]["number_of_repetitions"] > 1:
      stat_metrics = ["max", "min", "std", "avg"]
    else:
      stat_metrics = ["avg"]
    for metric in metrics:
        # replace underscores with spaces in the metric name
        print_metric = metric.replace("_", " ")
        # get parameters to be included in table
        param_list = get_default_param_list()
        if not "time" in metric:
            stat_metric = metric.split("_")[-1]
            param_list = [f"training_time_{stat_metric}"] + param_list
            if "test" in metric:  # add validation metric to table
                for temp_stat_metric in stat_metrics:
                    param_list = [f"final_val_{metric.split('_')[1]}_{temp_stat_metric}"] + param_list
            elif "val" in metric:  # add test metric to table
                param_list = [f"test_{metric.split('_')[2]}_{stat_metric}"] + param_list
        min_max = "min"  # partial headline for first iteration
        # treat the case where loss functions need to be compared without access to accuracy information by
        # adding the MAE loss to the table
        # if metric in min_max_lists[metric][0].keys[0].keys() and (metric != param_list[0]):
        if "loss" in metric:  # and (metric != param_list[0]):
            # param_list = [metric] + param_list
            param_list = [f"test_loss_mae_{stat_metric}"] + param_list
            first_line_label = f"\\textit{ {print_metric} }" + f" ({stat_metric})"
        else:
            first_line_label = r"\textit{" + print_metric + r"}"
        # top_n_list becomes min_list, then max_list
        for top_n_list in min_max_lists[metric]:
            # set stdout to a file for the current table
            original_stdout, file, dataset = set_print_filepath(
                metric, min_max, study_folder, expriment_name)
            # print table description
            if "accuracy" in metric:
                best_worst = "worst" if min_max == "min" else "best"
            else:
                best_worst = "best" if min_max == "min" else "worst"

            print_table_header(min_max, best_worst, top_n_list.len)

            # determine column widths for parameter column and for the remaining ones
            max_param_length = max([len(param) for param in param_list])
            list_max_value_length = max(
                [min(len(str(value)), prec+2) for value in top_n_list.values])
            # assemble and print line containing result values
            first_line = f"{first_line_label:{max_param_length}}"
            if "accuracy" in metric:
                value_iterator = list(reversed(top_n_list.values))
            else:
                value_iterator = top_n_list.values
            for value in value_iterator:
                # add values
                first_line += f" & {value:{list_max_value_length}.{prec}}"
            first_line += " \\\\"
            print(first_line)
            # assemble and print lines containing parameter settings
            for param in param_list:
                renamed_param = rename_param(param, param_setting_renaming)
                # add first column content = parameter names
                line = f"{renamed_param:{max_param_length}}"
                # get a list of the parameter settings
                param_settings = get_param_settings(top_n_list, param, metric)
                line = add_params_to_line(
                    line, param_settings, list_max_value_length, param_setting_renaming, prec=prec)
                # if "accuracy" in metric:
                #     print("", end="")
                #     pass
                # add color to line if all entries are equal
                if len(set(param_settings)) == 1:
                    line = color_line(line, color)
                line += " \\\\"
                print(line)
            tex_dataset = dataset.replace("_", " ")
            print("\\hline\n")
            print(
                f"\\caption{{{best_worst} settings regarding \\textit{{{print_metric}}} for the {tex_dataset} dataset}}")
            if label_prefix != "":
                print(
                    f"\\label{{table:{label_prefix}{metric}_{best_worst}_{dataset.lower()}}}")
            else:
                print(
                    f"\\label{{table:{metric}_{best_worst}_{dataset.lower()}}}")
            print("\\end{longtable}")
            min_max = "max"
            # reset stdout and close file
            sys.stdout = original_stdout
            file.close()
        print(f"saved min-max results for {print_metric}")


def print_table_header(min_max, best_worst, n_columns):
    """
    print table header and headline

    inputs:
    -------
        min_max - (str) -
        best_worst - (str) -
        top_n_list - (str) -
    """
    if best_worst == "best":  # color rightmost column with 'bestColumnColor'
        column_color = "bestColumnColor"
        # create and print latex column layout
        column_layout = '|l|' + \
            f'>{{\\columncolor{{{column_color}}}}}' + 'l|'*n_columns
    else:  # color leftmost column with 'worstColumnColor'
        column_color = "worstColumnColor"
        # create and print latex column layout
        column_layout = '|l|' + 'l|' * \
            (n_columns-1) + f'>{{\\columncolor{{{column_color}}}}}l|'
    print(f"\\begin{{longtable}}{{{column_layout}}}\n\\hline")
    # create and print table headline
    headline = r"\textbf{parameter name}"
    headline += f" & \\multicolumn{{{n_columns}}}{{c|}}{{\\textbf{{{best_worst} values}}}}"
    headline += " \\\\\n\\hline"
    print(headline)


def get_default_param_list():
    param_list = [  # list of parameter keys
        "neurons_per_layer",
        # "input_shape",
        # "output_shape",
        "activation_functions",
        "last_activation_function",
        # "layer_types",
        "loss_function",
        "training_data_percentage",
        "number_of_epochs",
        "batch_size",
        # "validation_split",
        # "number_of_repetitions",
        "optimizer",
        "learning_rate",
        "epsilon",
        # "beta_1",
        # "beta_2"
    ]
    return param_list


def add_params_to_line(line, param_settings, value_length, param_setting_renaming, prec=10):
    """
    add all values of `param_settings` to `line`, seperated by ' & ' and padded with spaces to reach length 'value_length'.
    float values in `param_settings` get rounded to `prec` decimal places.
    """
    for print_param in param_settings:
        if isinstance(print_param, (float, np.floating)):
            # print_param = round(print_param, prec)
            print_param = f"{print_param:.{prec}}"
        print_param = rename_param(print_param, param_setting_renaming)
        line += f" & {print_param:{value_length}}"
    return line


def color_line(line, color_name, seperator=" & "):
    """
    color a line of a latex table, where entries of different columns are seperated by `seperator`
    """
    column_entries = line.split(seperator)
    column_entries = [color_it(entry, color_name) for entry in column_entries]
    return seperator.join(column_entries)


def color_it(string, color_name):
    """
    add latex coloring to the given string
    """
    return f"{{\\color{{{color_name}}} {string.strip(' ')} }}"


def rename_param(value, renaming_dict):
    """
    rename the given value if it is in `renaming_dict`, otherwise replace underscores with spaces.
    return the new value
    """
    if value in renaming_dict:  # rename parameter setting before printing
        new_value = renaming_dict[value]
    else:  # convert underscores to spaces
        new_value = str(value).replace("_", " ")
    return new_value


def get_param_setting_renaming():
    """
    provide dictionary for renaming parameter settings
    Keys will be replaced by their corresponding values.
    """
    return {
        "mean_squared_error": "MSE",
        "categorical_crossentropy": "cat-cross",
        "log_cosh": "log cosh",
        "fast_c_adam": "fast cAdam",
        "fast_cAdam": "fast cAdam",
        "c_adam": "cAdam",
        "c_Adam": "cAdam",
        "my_adam": "my Adam",
        "adam": "Adam",
        "relu": "ReLU",
        "final_val_loss": "final validation loss",
        "final_val_accuracy": "final validation accuracy",
        "beta_1": "$\\beta_1$",
        "beta_2": "$\\beta_2$",
        "epsilon": "$\\varepsilon$",
        "1e-07": "$10^{-7}$",
        "1e+02": "100",
        "undetermined": "\\textit{unclear}"
    }


def get_param_settings(top_n_list, param, metric):
    """
    get the parameter settings of the given parameter for each entry of `top_n_list`

    inputs:
    -------
        top_n_list - (top_n_list) -
        param - (str) - a dictionary key for top_n_list.keys
    """
    param_settings = [setting[param] for setting in top_n_list.keys]
    if "accuracy" in metric:
        return list(reversed(param_settings))
    return param_settings


def set_print_filepath(metric, min_max, study_folder, expriment_name: str = None):
    """
    set the print output to the correct file for the current result calculation

    result tables will be saved in `study_folder/result_tables` as `.tex` files (one file for each table)
    """
    original_stdout = sys.stdout
    filepath = os.path.join(os.path.dirname(__file__),
                            "training_info", study_folder)
    # create result directory if it doesn't exist yet.
    if not "result_tables" in os.listdir(filepath):
        os.mkdir(os.path.join(filepath, "result_tables"))
    # determine the filename for the current table
    filename, dataset = get_filename(metric, min_max, study_folder, expriment_name)
    # complete filepath to the file for the current table
    filepath = os.path.join(filepath, "result_tables", filename)
    print(f"saving to {filepath}")
    # set stdout to the file
    file = open(filepath, "w")
    sys.stdout = file
    return original_stdout, file, dataset


def get_filename(metric, min_max, study_folder, experiment_name: str = None):
    """
    determine a filename for the current table
    """
    if min_max not in ("min", "max"):
        # filenames for parameter influence
        best_worst = min_max
    else:
        # filenames for min-max lists
        if "accuracy" in metric:
            min_max = "min" if min_max == "max" else "max"
        best_worst = "best" if min_max == "min" else "worst"
    if experiment_name is None:
      if "chem" in study_folder.lower():
          dataset = "ChemEx"
      elif "mnist" in study_folder.lower():
          dataset = "MNIST"
      else:
          dataset = "MNIST"
    else:
      dataset = experiment_name
    return f"{metric}_{dataset.lower()}_{best_worst}.tex", dataset


# if __name__ == "__main__":
#     print_best_worst_vertical(*args)
