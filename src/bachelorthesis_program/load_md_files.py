"""
convert parameter information saved as a markdown (.md) file to .json format.

Author: Sebastian Jost
"""
from helper_functions import build_filepath

def load_param_info(filename, study_folder=None, batch_folder=None, ending=".md"):
    """
    load parameter information from a .md file into a dictionary. all values get converted to their appropriate datatype.

    this function can load both study parameter files as well as batch parameter files
    """
    param_info = dict()
    optimizer_keys = list()
    add_optimizer_keys = False
    filepath = build_filepath(filename, study_folder, batch_folder, ending=ending)
    with open(filepath, "r") as file:
        for line in file.readlines():
            if line[:5] != "   - ":
                if "Optimizer" in line:
                    add_optimizer_keys = True
                continue # line containing no important information
            line = line[5:] # remove indentation
            line = line.strip() # remove leading and trailing whitespace
            key, raw_value = line.split(":")
            if add_optimizer_keys: # save keys of optimizer params
                optimizer_keys.append(key)
            value = get_value(raw_value.strip())
            param_info[key] = value
    return param_info, optimizer_keys



# def save_batch_param_json(filename, study_folder=None, batch_folder=None):
#     """
#     """


def get_value(raw_value):
    """
    convert a value given as a string to a fitting datatype
    """
    # lists
    if "[" in raw_value:
        if not "(" in raw_value:
            contents = raw_value[1:-1].split(",")
            return [get_value(sub_value.strip()) for sub_value in contents]
        # list of tuples
        contents = raw_value[1:-1].split(", (")
        return [get_value("(" + sub_value) for sub_value in contents]
    # tuple
    if "(" in raw_value:
        if not ",)" in raw_value:
            contents = raw_value.strip("()").split(",")
            return tuple(get_value(sub_value.strip()) for sub_value in contents)
        # singleton
        return (get_value(raw_value.strip("(),")),)
    # float
    if "." in raw_value or "e-" in raw_value[1:]:
        return float(raw_value)
    # int
    try:
        return int(raw_value)
    # str
    except ValueError:
        return raw_value.strip("'")


if __name__ == "__main__":
    # params, key_list = load_param_info("study_parameters", "bachelor_thesis_parameter_study_mnist")
    params, key_list = load_param_info("batch_parameters", "bachelor_thesis_parameter_study_mnist", "2021-09-10_15-20-25_batch")
    for key, value in params.items():
        print(key, ":", value, "\t", type(value))