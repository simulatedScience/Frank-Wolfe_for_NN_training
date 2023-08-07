"""
this module provides functions to output the results of an analysis of a parameter study to human readable form.

Author: Sebastian Jost
"""
import sys
import numpy as np
# try:
#     from sigfig import round
# except ModuleNotFoundError:
#     print("#"*60)
#     print("please install package 'sigfig' for better rounding")
#     print("#"*60)

from load_md_files import load_study_info
from file_management import load_parameter_analysis
from vertical_best_worst_output import print_best_worst_vertical, set_print_filepath, get_param_setting_renaming


def show_all_results(
        metrics=["training_time_avg", "test_loss_avg", "test_accuracy_avg"],
        filename="parameter_analysis_avg.pickle",
        prec: int = 5,
        study_folder: str = None,
        use_win_count: bool = False,
        label_prefix: str = "",
        experiment_name: str = None):
    """
    show results for the given metrics using data from the specified file.
    if study_folder is None, filename must be an absolute path leading to the analysis results file
    """
    # load study analysis results
    min_max_lists, param_count_results, param_win_ratios, param_diff_results, param_best_values = \
        load_parameter_analysis(filename=filename, study_folder=study_folder)
    print(f"results for {study_folder}")
    # show horizontal min_max lists
    # old_print_min_max_lists(min_max_lists, metrics)
    # show vertical min_max lists
    print_best_worst_vertical(
        min_max_lists,
        metrics,
        prec=prec,
        study_folder=study_folder,
        label_prefix=label_prefix,
        experiment_name=experiment_name)
    print()
    print("-"*60)
    print()
    # show results for best parameter values for each metric
    if use_win_count:
        print_param_influcence_tables(
            param_count_results,
            param_diff_results,
            param_best_values,
            metrics,
            study_folder,
            use_win_count,
            label_prefix=label_prefix,
            expriment_name=experiment_name)
    else:
        print_param_influcence_tables(
            param_win_ratios,
            param_diff_results,
            param_best_values,
            metrics,
            study_folder,
            use_win_count,
            label_prefix=label_prefix,
            expriment_name=experiment_name)


def print_param_influcence_tables(param_count_results,
                                  param_diff_results,
                                  param_best_values,
                                  metrics,
                                  study_folder,
                                  use_win_count,
                                  prec=3,
                                  label_prefix="",
                                  expriment_name: str = None):
    """
    print table showing the results of the comparison between different values for each parameter
    print parameter influence tables
    """
    param_setting_renaming = get_param_setting_renaming()
    # load study parameters
    study_settings, optimizer_params_list = load_study_info(
        filename="study_parameters", study_folder=study_folder)
    for metric in metrics:
        # determine filename
        if use_win_count:
            fileending = "count"
        else:
            fileending = "ratios"
        original_stdout, file, dataset = set_print_filepath(
            metric, fileending, study_folder, expriment_name)
        print(f"% best parameter values regarding \\texttt{{{metric}}}")
        print_headline(metric, fileending)
        for dict_item in study_settings.items():
            param_name, param_values = dict_item
            if not isinstance(param_values, list):
                continue
            param_values = cell_entry_to_string(param_values)
            # win_count = cell_entry_to_string(param_count_results[param_name][metric])
            win_count = win_ratio_to_string(
                param_count_results[param_name][metric])
            win_count = cell_entry_to_string(win_count)
            # avg_diff_values = cell_entry_to_string(np.round(param_diff_results[param_name][metric], prec))
            avg_diff_values = avg_diff_to_string(
                param_diff_results[param_name][metric], prec)
            avg_diff_values = cell_entry_to_string(avg_diff_values)
            best_value = str(param_best_values[param_name][metric])
            param_name = param_name.replace("_", " ")
            print_line(param_name, param_values, win_count, avg_diff_values,
                       best_value, renaming_dict=param_setting_renaming)
        print_table_end(metric, fileending, dataset, label_prefix=label_prefix)
        print()
        # reset stdout and close file
        sys.stdout = original_stdout
        file.close()
        print(f"saved win-{fileending} results for {metric}")


def print_headline(metric=None, count_or_ratio="ratios"):
    """
    print a table headline for latex
    """
    if count_or_ratio == "ratios":
        count_or_ratio += r" in \%"
    # if "time" in metric:
    #     unit = " in s"
    # else:
    #     unit = ""
    diff_unit = r" in \%"
    column_titles = ["parameter name",
                     "parameter values",
                     f"win {count_or_ratio}",
                     f"avg. differences{diff_unit}",
                     "best value"]
    for i, title in enumerate(column_titles):
        column_titles[i] = f"\\textbf{{{title}}}"
        if i == 0 or i == len(column_titles)-1:
            continue
        # make multicolumn
        column_titles[i] = f"\\multicolumn{{{6}}}{{c|}}{{{column_titles[i]}}}"

    column_format = '|l|' + 'c|'*6*(len(column_titles)-2) + "c|"
    print(f"\\begin{{longtable}}{{{column_format}}}")
    print(r"\hline")
    # assemble and print headline
    seperator = " & "
    line = seperator.join(column_titles)
    # add line ending and print headline
    print(line + r" \\" + "\n" + r"\hline")


def print_line(*column_contents, renaming_dict=None):
    """
    print a table line for latex
    """
    if renaming_dict is not None:
        column_contents = replace_in_line(renaming_dict, *column_contents)
    seperator = " & "
    line = seperator.join(column_contents)
    print(line + r" \\")


def print_table_end(metric, fileending="ratios", dataset="", label_prefix=""):
    """
    print the end of a latex table
    """
    print("\\hline\n")
    print_metric = metric.replace('_', ' ')
    tex_dataset = dataset.replace('_', ' ')
    print(
        f"\\caption{{parameter influence regarding \\textit{{{print_metric}}} for the {tex_dataset} dataset}}")
    if label_prefix != "":
        print(
            f"\\label{{table:{label_prefix}{metric}_{fileending}_{dataset.lower()}}}")
    else:
        print(f"\\label{{table:{metric}_{fileending}_{dataset.lower()}}}")
    print("\\end{longtable}")


def replace_in_line(renaming_dict, *column_contents):
    """
    in `column_contents` replace any keys in `renaming_dict` with their values
    """
    if isinstance(column_contents, tuple):
        column_contents = list(column_contents)
    for i, column_elem in enumerate(column_contents):
        for key, value in renaming_dict.items():
            column_elem = column_elem.replace(key, value)
        column_contents[i] = column_elem
    return column_contents


def cell_entry_to_string(value_list, total_width=6):
    """
    convert a cell entry given as a list or numpy array to a string for a latex table.

    inputs:
    -------
        value - (list) - any list, tuple or numpy array of values that supports `len(value_list)`
        total_width - (int) - lowest common multiple of the lengths of all lists in the column.
    """
    n = len(value_list)
    cell_width = total_width//n
    new_entries = [""]*n
    for i, entry in enumerate(value_list):
        linetype = ":" if i < total_width-cell_width-2 else "|"
        new_entries[i] = f"\\multicolumn{{{cell_width}}}{{c{linetype}}}{{{entry}}}"
    return " & ".join(new_entries)


def win_ratio_to_string(win_ratios, prec=1):
    """
    round list of win ratios to given precision for printing with reasonable accuracy.
    automatically detect if win counts (integers) are given and don't change those.

    inputs:
    -------
        win_ratios - (list) - list of ints or floats
        prec - (int) - number of significant figures
    """
    if isinstance(win_ratios[0], int):  # values are counts
        return win_ratios
    percentages = [100*elem for elem in win_ratios]
    return avg_diff_to_string(percentages, prec)


def avg_diff_to_string(avg_diffs, prec=3):
    """
    round list of floats to `prec` significant figures

    inputs:
    -------
        win_ratios - (list) - list of ints or floats
        prec - (int) - number of significant figures
    """
    str_avg_diffs = ["" for _ in avg_diffs]
    for i, elem in enumerate(avg_diffs):
        if elem == 0:  # convert 0.0 to 0
            str_avg_diffs[i] = f"{0:{prec+2}}"
        else:
            str_avg_diffs[i] = str(round(elem, prec))  # round using sigfig
    return str_avg_diffs


if __name__ == "__main__":
    # file dialog to choose study_folder
    from tkinter import filedialog
    # study_folder = filedialog.askdirectory(initialdir=".", title="Select study folder")
    # study_folder = filedialog.askdirectory(title="Select study folder")
    # study_folder = study_folder.split("/")[-1]

    # mnist
    study_folder = "C:\\future_D\\uni\\Humboldt Uni\\Nebenhoerer SoSe 2023\\FW_NN_training\\src\\bachelorthesis_program\\training_info\\2023-08-07_00-24-41_mnist_sfw_parameter_study"
    # study_folder = "2021-09-09_21-56-12_parameter_study_debug"

    # # chemex
    # study_folder = "bachelor_thesis_parameter_study_chem"
    # metrics = ["training_time_avg", "final_val_loss_mae_avg", "test_loss_avg"]
    metrics = ["training_time_avg", "test_loss_avg", "test_accuracy_avg"]
    # metrics = ["training_time_avg", "test_loss_avg"]
    # metrics = ["final_val_loss_mae_avg"]
    show_all_results(metrics=metrics, study_folder=study_folder, filename="parameter_analysis_avg.pickle", experiment_name="sfw_mnist")
    # show_all_results(metrics=metrics, study_folder=study_folder, filename="parameter_analysis_avg.pickle", experiment_name="chemReg_Adam")
    # show_all_results(metrics=metrics, study_folder=study_folder, filename="parameter_analysis_std.pickle", expriment_name="chemReg_Adam_std")
    # show_all_results(metrics=metrics, study_folder=study_folder, filename="parameter_analysis_min.pickle", expriment_name="chemReg_Adam_min")
    # show_all_results(metrics=metrics, study_folder=study_folder, filename="parameter_analysis_max.pickle", expriment_name="chemReg_Adam_max")

    # # fast cAdam
    # study_folder = "bachelor_thesis_param_study_fastcAdam"
    # metrics = ["training_time", "test_accuracy"]
    # show_all_results(
    #     metrics=metrics, study_folder=study_folder, label_prefix="fast_")

    # # Adam variant comparison
    # study_folder = "bachelor_thesis_adam_variants"
    # metrics = ["training_time", "test_accuracy"]
    # show_all_results(
    #     metrics=metrics, study_folder=study_folder, label_prefix="variant_")
    # study_folder = "adam_variants_with_sgd"
    # study_folder = "adam_variants_with_fcadam"
    # metrics = ["training_time_avg", "test_accuracy_avg"]
    # show_all_results(
    #     metrics=metrics, study_folder=study_folder, label_prefix="variant_", expriment_name="cadam_variants")

    # study_folder = "cadam_speed_study"
    # metrics = ["training_time_avg", "test_accuracy_avg"]
    # show_all_results(
    #     metrics=metrics, study_folder=study_folder, label_prefix="variant_", expriment_name="cadam_speed")
    # ML paper chemReg cAdam study
    # study_folder = "ML_paper_chemReg_cAdam_study"
    # # metrics = ["training_time", "test_loss"]
    # metrics = ["final_val_loss_mae_avg"]
    # show_all_results(metrics=metrics, study_folder=study_folder, filename="parameter_analysis_avg.pickle", expriment_name="chemReg_cAdam")
