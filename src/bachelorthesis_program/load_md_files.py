"""
convert parameter information saved as a markdown (.md) file to .json format.

Author: Sebastian Jost
"""
from helper_functions import build_filepath

def load_batch_info(filename, study_folder=None, batch_folder=None, ending=".md"):
    """
    load parameter information from a .md file into a dictionary. all values get converted to their appropriate datatype.

    this function can load both study parameter files as well as batch parameter files
    """
    param_info = dict()
    optimizer_keys = list()
    # optimizer_params_list = list()
    # optimizer_setup_index = -1
    add_optimizer_keys = False
    filepath = build_filepath(filename, study_folder, batch_folder, ending=ending)
    with open(filepath, "r") as file:
        for line in file.readlines():
            if line[:5] != "   - ":
                if "### Optimizer" in line:
                    add_optimizer_keys = True
                # if "optimizer setup" in line:
                #     optimizer_setup_index += 1
                #     optimizer_params_list.append(dict())
                continue # line containing no important information
            line = line[5:] # remove indentation
            line = line.strip() # remove leading and trailing whitespace
            key, *raw_value = line.split(":")
            if len(raw_value) > 1:
                raw_value = ":".join(raw_value)
            else:
                raw_value = raw_value[0]
            value = get_value(raw_value.strip())
            if add_optimizer_keys: # save keys of optimizer params
                optimizer_keys.append(key)
                # optimizer_params_list[optimizer_setup_index][key] = value
            param_info[key] = value
    return param_info, optimizer_keys


def load_study_info(filename, study_folder=None, batch_folder=None, ending=".md"):
    """
    load parameter information from a .md file into a dictionary. all values get converted to their appropriate datatype.

    this function can load both study parameter files as well as batch parameter files
    """
    param_info = dict()
    filepath = build_filepath(filename, study_folder, batch_folder, ending=ending)
    with open(filepath, "r") as file:
        for line in file:
            if line[:5] != "   - ":
                if "### Optimizer" in line:
                    break
                continue # line containing no important information
            line = line[5:] # remove indentation
            line = line.strip() # remove leading and trailing whitespace
            key, raw_value = line.split(":")
            value = get_value(raw_value.strip())
            param_info[key] = value

        # read optimizer parameters
        optimizer_params_list = list()
        optimizer_setup_index = -1
        for line in file: # loop starts where previous loop ended
            if line[:5] != "   - ":
                if "optimizer setup" in line:
                    optimizer_setup_index += 1
                    optimizer_params_list.append(dict())
                continue
            line = line[5:] # remove indentation
            line = line.strip() # remove leading and trailing whitespace
            optimizer_key, raw_value = line.split(":")
            value = get_value(raw_value.strip())
            optimizer_params_list[optimizer_setup_index][optimizer_key] = value
        param_info["optimizer_params"] = optimizer_params_list
    return param_info, optimizer_params_list

# def save_batch_param_json(filename, study_folder=None, batch_folder=None):
#     """
#     """


def get_value(raw_value):
    """
    convert a value given as a string to a fitting datatype
    """
    # dicts
    if "{" in raw_value:
        contents = raw_value[1:-1].split(", ")
        return {sub_value.split(":")[0].strip():
                get_value(sub_value.split(":")[1].strip()) for sub_value in contents
            }
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
    params, key_list = load_batch_info("batch_parameters", "bachelor_thesis_parameter_study_mnist", "2021-09-10_15-20-25_batch")
    for key, value in params.items():
        print(key, ":", value, "\t", type(value))