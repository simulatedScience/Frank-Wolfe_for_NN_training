"""
this module implements a class used to store the best or worst n elements of a parameter study.
each list contains at most n key-value pairs, sorted by their value.

Author: Sebastian Jost
"""
import numpy as np

class top_n_list:
    """
    a list saving the best or worst n settings.
    Each entry should be a key-value pair.

    The list is kept sorted (low to high value)
    The min and max value of the list are saved for fast comparison.

    A method allows easily adding an element to the sorted list.

    A method allows detecting differences in the keys of the list.

    Method: print values and keys as table, differences in the keys are marked with color.
    """
    def __init__(self, n_elems=5, mode="min"):
        self.mode = mode.lower()
        self.len = n_elems
        self.min_value = np.inf if mode=="min" else -np.inf
        self.max_value = np.inf if mode=="min" else -np.inf
        self.keys = list()
        self.values = list()

    def add_item(self, key_dict, value, ignore_keys):
        """
        insert a key, value pair into the sorted key and value lists
            while keeping the length the same (= dropping one value if necessary)
        if value is below the current minimum value (mode max) or above the maximum value (mode min), nothing happens
        for the print method, the key should be a dictionary containing parameter settings that yield the given value otherwise the class works for any key datatype
        """
        # if value in self.values and value in self.values[1:]:
        #     print("known value")
        if self.duplicate_key(key_dict, ignore_keys):
            return
        if self.mode == "min":
            if value < self.max_value:
                if len(self.values) == 0:
                    self.keys.append(key_dict)
                    self.values.append(value)
                else:
                    for i, old_value in enumerate(self.values):
                        if value < old_value:
                            break
                    if len(self.values) < self.len:
                        self.keys = self.keys[:i] + [key_dict] + self.keys[i:]
                        self.values = self.values[:i] + [value] + self.values[i:]
                    else:
                        # drop last element
                        self.keys = self.keys[:i] + [key_dict] + self.keys[i:-1]
                        self.values = self.values[:i] + [value] + self.values[i:-1]
            elif len(self.values) < self.len-1:
                self.keys.append(key_dict)
                self.values.append(value)
            else:
                return
        elif self.mode == "max":
            if value > self.min_value:
                if len(self.values) == 0:
                    self.keys.append(key_dict)
                    self.values.append(value)
                else:
                    for i, old_value in enumerate(self.values):
                        if value < old_value:
                            i -= 1
                            break
                    if len(self.values) < self.len:
                        self.keys = self.keys[:i+1] + [key_dict] + self.keys[i+1:]
                        self.values = self.values[:i+1] + [value] + self.values[i+1:]
                    else:
                        # drop first element
                        self.keys = self.keys[1:i+1] + [key_dict] + self.keys[i+1:]
                        self.values = self.values[1:i+1] + [value] + self.values[i+1:]
            elif len(self.values) < self.len-1:
                self.keys.append(key_dict)
                self.values.append(value)
            else:
                return
        self.min_value = self.values[0]
        self.max_value = self.values[-1]


    def duplicate_key(self, new_key, ignore_keys=None):
        """
        check if `new_key` is in `self.keys`. If `new_key` is a dictionary, the keys given in `ignore_keys` are not compared.
        That means two dictionaries `new_key` and one in `self.keys` are considered equal if they are equal for all keys except `ignore_keys`.

        inputs:
        -------
            new_key - (any) - any key that may be saved in the `top_n_list`.
            ignore_keys - (list) of (any) - list of any keys that might occur in `new_key`.
                if `new_key` is not a dictionary, this argument gets ignored.

        outputs:
        --------
            (bool) - `True` if `new_key` is in `self.keys`, otherwise `False`, note the special behavior described above
        """
        if new_key in self.keys:
            return True
        if not isinstance(new_key, dict):
            return False
        for existing_key in self.keys:
            for dict_key, value in existing_key.items():
                if dict_key in ignore_keys:
                    continue
                if not dict_key in new_key.keys() or value != new_key[dict_key]:
                    # `new_key` differs from `existing_key` in an entry that's not ignored.
                    break
            else:
                # `new_key` is considered equal to at least one entry of `self.keys`
                return True
        return False




    def print_values_keys(self, key_list):
        """
        print list as value-key pairs.
        """
        value_length = max([len(str(value)) for value in self.values])
        key_lengths = [max([len(str(key_dict[key])) for key_dict in self.keys]) for key in key_list]
        # update column widths to fit the headline labels
        value_length = max(value_length, len("value"))
        key_lengths = [max(len(str(key)), key_length) for key, key_length in zip(key_list, key_lengths)]
        # assemble headline
        headline = f"| {'value':>{value_length}} |"
        for key_length, key_label in zip(key_lengths, key_list):
            headline += f" {key_label:>{key_length}} |"
        print(headline)
        print("|" + "---|"*(len(key_list)+1))
        # print table contents (value, key pairs)
        for key_dict, value in zip(self.keys, self.values):
            line = f"| {value:>{value_length}} |"
            for key, key_length in zip(key_list, key_lengths):
                line += f" {str(key_dict[key]):>{key_length}} |"
            print(line)
        print()
