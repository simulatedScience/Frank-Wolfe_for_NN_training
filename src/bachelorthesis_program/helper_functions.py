"""
this module implements some generally useful functions to be used by other modules

Author: Sebastian Jost
"""
import os
import time
import itertools
from tensorflow.config import list_physical_devices

def get_date_time(date_time_format="%Y-%m-%d_%H-%M-%S"):
    """
    return current date and time as a string: `'y-m-d_H-M-S'`
    """
    return time.strftime(date_time_format)


def auto_use_multiprocessing():
    """
    automatically determine whether or not to use CPU based multiprocessing for training models.

    multiprocessing is not used if at least one GPU is set up for use with tensorflow.
    """
    n_gpus = len(list_physical_devices('GPU'))
    if n_gpus > 0:
        return False
    return True


def get_max_key_length(*dicts):
    """
    determine the maximum length of a key in any of the given dictionaries.
    the keys of each dict must allow being passed to the built-in `len` function.
    """
    max_len = 0
    for temp_dict in dicts:
        for key in temp_dict.keys():
            if len(key) > max_len:
                max_len = len(key)
    return max_len


def build_filepath(filename, study_folder=None, batch_folder=None, ending=".pickle"):
    """
    convert a filename with foldernames to an absolute filepath
    this is specifally meant to save or load training info files for my neural network analysis program
    """
    # check whether the given filename is already an absolute path
    if os.path.dirname(__file__) in filename:
        return filename
    # otherwise build path with the given folders and filename and ending
    filepath = os.path.join(os.path.dirname(__file__), "training_info")
    if study_folder is not None:
        filepath = os.path.join(filepath, study_folder)
    if batch_folder is not None:
        filepath = os.path.join(filepath, batch_folder)
    if not ending in filename:
        filename += ending
    filepath = os.path.join(filepath, filename)
    return filepath


def dict_cross_product(*dictionaries):
    """
    return a generator that outputs every combination of the values in the given dictionaries.
    the keys of the dicts should be disjoint
    values that are lists will be split, other values remain constant for all combinations

    inputs:
    -------
        any number of dictionaries
    """
    input_lists = list()
    for input_dict in dictionaries:
        input_lists += input_dict.values()
    input_lists = [x if isinstance(x, list) else [x] for x in input_lists]
    key_list = list()
    for input_dict in dictionaries:
        key_list += input_dict.keys()
    product_generator = itertools.product(*input_lists)
    return product_generator, key_list