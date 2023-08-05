This directory includes several modules working together to simplify training some neural networks and then save and compare information about the training process.


## What the program does

For more details, see the file `Bachelorthesis.pdf` at (https://github.com/simulatedScience/Bachelorthesis)[GitHub.com/simulatedScience]. Compared to the code prvoided there, this version changes the result analysis by using more relative measures and adjusting the output tables accordingly.
Due to the different goals of the experiments, the training on the chemReg dataset was repeated, including cAdam. See `dense_parameter_study_chemex_cadam.py` for details on that.
Additionally, other experiments regarding the runtime performance of cAdam were repeated. See `dense_parameter_study_mnist_cadam_speed.py` for the runtime comparison experiment.
The paper also claims that the intuitive variant of cAdam to use the bias corrected $\hat{m}_t$ is much worse. This is backed by the experiment in `dense_parameter_study_mnist_cadam_variants.py`.

A summary of the produced result tables can be found in the subfolder `results`.

## how to use the program
0. to check that everything is working, run `install_test.py` (this should take at most a few minutes)
1. choose some possible values for each tunable parameter
   (see `dense_parameter_study_[...].py` files)
2. train model several times for every combination of the parameters chosen in 1. (this process is started in `dense_parameter_study_[...].py` files using the method `dense_parameter_study` in `dense_parameter_study.py`)
3. examine the results of the tests in two steps:
   1. summarize all results of the trained models and calculate averages of models that were trained multiple times. (using the methods `statistical_analysis_of_study` and `analyse_parameter_study` in `data_analysis.py`)
   2. Convert the results to latex tables. (using `show_all_results` in `result_ouput.py`)

## Notes
Most parts of the program are kept quite general and expandable, however some parts of the result output are specialized to the experiments performed for the thesis.

For example `vertical_best_worst_output.py` includes a method `get_param_setting_renaming` defining a dictionary that specifies some table entries to get replaced with other strings (mostly to add formatting or fix spelling).

The method `print_param_influcence_tables` in `result_output.py` assumes that the number of values for each parameter is always a factor of 6. This 6 should actually be the lowest common multiple of all numbers of values for the parameters.

It's likely that there are a few similar issues that require a bit of work before using this program for other studies. It is also possible to combine the steps 2. and 3.1 of the workflow described above (automatically start step 3.1 after 2.).

## datasets

The MNIST dataset used for most of the experiments is included with Tensorflow and freely available online.

The other experiment uses the dataset contained in the folder `chem_data`. Both the data and code to read it were provided to me by Franz Bethke. Authors are given in the corresponding files.