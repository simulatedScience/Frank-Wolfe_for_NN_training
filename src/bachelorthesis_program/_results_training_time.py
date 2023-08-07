import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from load_md_files import load_batch_info

def create_training_time_table(study_folder, error_threshold=300):
    # Dictionary to store the table data
    table_data = {'Optimizer Configuration': []}
    
    history_groups = group_batches(study_folder)
    for hidden_layers, batch_folder_paths in history_groups.items():
        # Create a column for each hidden layer configuration
        table_data[f'Hidden Layers {hidden_layers}'] = []

        for folder_path in batch_folder_paths:
            histories = load_histories(folder_path)
            n_epochs, label, _ = get_batch_info(folder_path)

            # Add optimizer configuration to 'Optimizer Configuration' column
            if label not in table_data['Optimizer Configuration']:
                table_data['Optimizer Configuration'].append(label)

            # Calculate average training time for the batch
            training_time = np.mean([history_obj.train_time for history_obj in histories])

            # Replace time value with NaN if it's greater than error_threshold
            training_time = np.nan if training_time > error_threshold else training_time

            # Add training time to the corresponding hidden layer column
            table_data[f'Hidden Layers {hidden_layers}'].append(training_time)
    return table_data


def plot_bar_chart(table_data):
    table_data = sort_table_data(table_data)

    # Number of optimizer configurations
    n_optimizers = len(table_data['Optimizer Configuration'])

    # Number of hidden layer architectures
    n_hidden_layers = len(table_data) - 1

    # Bar width
    bar_width = 0.15

    # X-axis positions for each group of bars
    r = np.arange(n_optimizers)

    # Plot the bars for each hidden layer architecture
    for i, hidden_layers in enumerate([key for key in table_data.keys() if key != 'Optimizer Configuration']):
        plt.bar(r + i * bar_width, table_data[hidden_layers], width=bar_width, label=hidden_layers)

    # Set the X-axis tick labels
    plt.xticks(r + bar_width * (n_hidden_layers - 1) / 2, table_data['Optimizer Configuration'], rotation=45, ha='right')

    plt.ylabel('training time in s')
    plt.title('Training time for different optimizer configurations')
    plt.legend()
    plt.grid(color="#dddddd", axis='y')
    plt.tight_layout()
    plt.show()

def sort_table_data(table_data):
    # Extract the last hidden layer configuration
    last_hidden_layers_key = list(table_data.keys())[-1]

    # Create a list of tuples with optimizer configuration and corresponding training time for the last hidden layer
    sorted_data = sorted(zip(table_data['Optimizer Configuration'], *[table_data[key] for key in table_data.keys() if key != 'Optimizer Configuration']), key=lambda x: x[-1])

    # Reorganize the sorted data into the table_data format
    sorted_table_data = {
        'Optimizer Configuration': [item[0] for item in sorted_data]
    }
    for i, key in enumerate([key for key in table_data.keys() if key != 'Optimizer Configuration']):
        sorted_table_data[key] = [item[i + 1] for item in sorted_data]

    return sorted_table_data
def load_histories(folder_path):
    histories = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pickle') and "history" in filename:
            with open(os.path.join(folder_path, filename), 'rb') as file:
                history = pickle.load(file)
                histories.append(history)
    return histories

def get_batch_info(batch_folder_path: str, metric: str = "val_loss"):
    """
    Simplify training history to include datapoints per epoch rather than per batch

    Args:
      history_obj: history object of keras model
    """
    # load batch info
    param_info, _ = load_batch_info(filename=os.path.join(batch_folder_path, "batch_parameters.md"))
    n_epochs = param_info["number_of_epochs"]
    hidden_layers = param_info["neurons_per_layer"]
    try:
        label = param_info["optimizer_params"]
        if label["optimizer_type"].lower() in ("adam", "sgd"):
            label = label["optimizer_type"]
            if "weight_decay" in label:
                label += f"+wd={label['weight_decay']}"
        elif label["constraints_type"].lower() in ("ksparse", "knorm"):
            label = f'{label["optimizer_type"]}+{label["constraints_type"]}+{label["constraints_K"]}'
        else:
            label = f'{label["optimizer_type"]}+{label["constraints_type"]}'
    except KeyError:
        label = param_info["optimizer"]
    # print(f"Optimizer: {label}, hidden layers: {hidden_layers}")
    return n_epochs, label, hidden_layers

def group_batches(study_folder):
    history_groups = dict()
    for batch_folder in os.listdir(study_folder):
        if not os.path.isdir(os.path.join(study_folder, batch_folder)):
            continue
        if "result_tables" in batch_folder:
            continue
        batch_folder_path = os.path.join(study_folder, batch_folder)
        n_epochs, label, hidden_layers = get_batch_info(batch_folder_path)
        if not hidden_layers in history_groups:
            history_groups[hidden_layers] = [batch_folder_path]
        else:
            history_groups[hidden_layers].append(batch_folder_path)
    return history_groups


def create_dataframe(table_data):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(table_data)
    return df

def main(study_folder, error_threshold=300):
    table_data = create_training_time_table(study_folder, error_threshold=error_threshold)
    df = create_dataframe(table_data)
    print(df)
    plot_bar_chart(table_data)

if __name__ == "__main__":
    # MNIST diagrams #1 (32, 64)
    study_folder = "C:\\future_D\\uni\\Humboldt Uni\\Nebenhoerer SoSe 2023\\FW_NN_training\\src\\bachelorthesis_program\\training_info\\2023-08-07_10-43-21_mnist_reg_sfw_parameter_study"
    main(study_folder, error_threshold=300)

    # ChemReg diagrams #1 ((64, 16), (32, 32))
    study_folder = "C:\\future_D\\uni\\Humboldt Uni\\Nebenhoerer SoSe 2023\\FW_NN_training\\src\\bachelorthesis_program\\training_info\\2023-08-07_15-30-46_chemreg_reg_sfw_parameter_study"
    main(study_folder, error_threshold=300)