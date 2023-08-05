"""
This module contains the `parse_chem_data` function.
"""
from pandas import DataFrame
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler


def parse_chem_data(filename):
    """
    Reads contents of arff file `filename` to generate input/output values
    for a learning problem.

    The input file is assumed to define a single relation whose last value
    should be learned from the others. The input values for this functional
    mapping are preprocessed to be normally distributed.

    Parameters
    ----------
    filename: str
        Path to the arff data file that should be parsed.

    Returns
    -------
    numpy.ndarray
        Inputs to the functional relation that will later be learned.
    numpy.ndarray
        Outputs of the functional relation.

    # This function was provided by F. Bethke.
    """

    with open(filename, 'r') as data_file:
        raw_data, meta_data = loadarff(data_file)

    data = DataFrame(raw_data, columns=meta_data.names()).values

    inputs = data[:, :-1].astype(float)
    outputs = data[:, -1].astype(float)
    # preprocessing is apparently necessary (see [Ren19] section 2.1.4)
    inputs = StandardScaler().fit_transform(inputs)

    return inputs, outputs


def get_scaler(filename):
    """
    get the parameters used for scaling the training data. This needs to be applied to any input the network is supposed to evaluate.

    inputs:
    -------
        filename - (str) - Path to the arff data file that should be parsed.
    """
    with open(filename, 'r') as data_file:
        raw_data, meta_data = loadarff(data_file)

    data = DataFrame(raw_data, columns=meta_data.names()).values

    inputs = data[:, :-1].astype(float)
    scaler = StandardScaler()
    scaler = scaler.fit(inputs)
    return scaler


def test_inputs(filename="features_augmented_05percent.arff"):
    inputs, outputs = parse_chem_data(filename)
    for i, x in enumerate(inputs):
        if x[9] != x[10]:
            print(f"{i:5}", "\t", x[9], x[10])
        else:
            print(f"equal: {i:5} \t {x[9]} == {x[10]}")


if __name__ == "__main__":
    test_inputs()
