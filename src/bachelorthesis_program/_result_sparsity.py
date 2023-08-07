import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from load_md_files import load_batch_info


def count_nonzero_params(model, threshold=0.0):
    nonzero_count = 0
    total_count = 0
    max_param_value = 0

    for param in model.parameters():
        total_count += torch.numel(param)
        max_param_value = max(max_param_value, torch.max(torch.abs(param)).item())
        # convert to numpy array
        param = param.detach().cpu().numpy()
        # count nonzero params
        nonzero_count += np.count_nonzero(param > threshold)
    # print(f"max param value: {max_param_value}")
    sparsity = 1 - nonzero_count / total_count
    # zero_count = total_count - nonzero_count
    return nonzero_count, sparsity

def eval_batch_sparsity(batch_folder, threshold=0.0):
    # iterate over all .pt files
    sparsity_sum = 0
    param_count_sum = 0
    n_models = 0
    for filename in os.listdir(batch_folder):
        if filename.endswith('.pt'):
            # load model
            model = torch.load(os.path.join(batch_folder, filename))
            # count nonzero params
            nonzero_count, sparsity = count_nonzero_params(model, threshold)
            # print(f'{filename}: {nonzero_count} nonzero params, {sparsity*100}% sparsity')
            # add to sum
            sparsity_sum += sparsity
            param_count_sum += nonzero_count
            n_models += 1
    avg_sparsity = sparsity_sum / n_models
    avg_param_count = param_count_sum / n_models
    batch_name, hidden_neurons = get_batch_name(batch_folder)
    return avg_sparsity, avg_param_count, batch_name

def get_batch_name(batch_folder_path):
    param_info, _ = load_batch_info(filename=os.path.join(batch_folder_path, "batch_parameters.md"))
    
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
    return label, hidden_layers

def group_batches(study_folder):
    history_groups = dict()
    for batch_folder in os.listdir(study_folder):
        if not os.path.isdir(os.path.join(study_folder, batch_folder)):
            continue
        if "result_tables" in batch_folder:
            continue
        batch_folder_path = os.path.join(study_folder, batch_folder)
        label, hidden_layers = get_batch_name(batch_folder_path)
        if not hidden_layers in history_groups:
            history_groups[hidden_layers] = [batch_folder_path]
        else:
            history_groups[hidden_layers].append(batch_folder_path)
    return history_groups

def print_group_sparsities(folder_paths, title_addon="", threshold=0.0):
    batch_names = []
    avg_sparsities = []
    avg_param_counts = []
    
    for folder_path in folder_paths:
        avg_sparsity, avg_param_count, batch_name = eval_batch_sparsity(folder_path, threshold=threshold)
        print(f"{batch_name} uses on average {avg_param_count:.0f} parameters = average sparsity of {avg_sparsity*100:.1f}%")
        
        batch_names.append(batch_name)
        avg_sparsities.append(avg_sparsity * 100)  # Sparsity in %
        avg_param_counts.append(avg_param_count)

    # Sort by avg_sparsity
    sorted_data = sorted(zip(batch_names, avg_sparsities, avg_param_counts), key=lambda x: x[1])
    batch_names, avg_sparsities, avg_param_counts = zip(*sorted_data)

    # Create a figure and axis object
    fig, ax1 = plt.subplots()

    # Create the bar chart with sparsity percentages
    ax1.bar(batch_names, avg_sparsities, color='blue', alpha=0.5)
    ax1.set_xlabel('optimizer configuration')
    ax1.set_ylabel('Sparsity (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    # Rotate x-tick labels for better visibility
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # Create a second y-axis to show the exact number of nonzero parameters
    ax2 = ax1.twinx()
    color2 = "#dd7700"
    ax2.plot(batch_names, avg_param_counts, color=color2, marker='o', linestyle="")
    ax2.set_ylabel('number of nonzero parameters', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(bottom=0)



    plt.title('Sparsity Measurements for different optimizers' + title_addon)
    plt.tight_layout()
    plt.show()

def main(study_folder, threshold=0.0):
    history_groups = group_batches(study_folder)
    for hidden_layers, batch_folder_paths in history_groups.items():
        print("#"*50)
        print(f"Hidden layers: {hidden_layers}")
        print_group_sparsities(batch_folder_paths, title_addon=f"\nHidden layers: {hidden_layers}", threshold=threshold)

if __name__ == '__main__':
    import tkfilebrowser
    # let user choose several folders to plot
    # folder_paths = tkfilebrowser.askopendirnames(title='Choose batch folders to plot')

    # print_group_sparsities(folder_paths, threshold=100000000)

    threshold = 1e-12
    # # MNIST diagrams #0 (32, 64)
    # study_folder = "C:\\future_D\\uni\\Humboldt Uni\\Nebenhoerer SoSe 2023\\FW_NN_training\\src\\bachelorthesis_program\\training_info\\2023-08-07_00-24-41_mnist_sfw_parameter_study"
    # main(study_folder, threshold=threshold)

    # MNIST diagrams #1 (32, 64)
    study_folder = "C:\\future_D\\uni\\Humboldt Uni\\Nebenhoerer SoSe 2023\\FW_NN_training\\src\\bachelorthesis_program\\training_info\\2023-08-07_10-43-21_mnist_reg_sfw_parameter_study"
    main(study_folder, threshold=threshold)

    # # MNIST diagrams #2 (8, 16)
    # study_folder = "C:\\future_D\\uni\\Humboldt Uni\\Nebenhoerer SoSe 2023\\FW_NN_training\\src\\bachelorthesis_program\\training_info\\2023-08-07_12-43-04_mnist_reg_sfw_2_parameter_study"
    # main(study_folder, threshold=threshold)

    # ChemReg diagrams
    study_folder = "C:\\future_D\\uni\\Humboldt Uni\\Nebenhoerer SoSe 2023\\FW_NN_training\\src\\bachelorthesis_program\\training_info\\2023-08-07_15-30-46_chemreg_reg_sfw_parameter_study"
    main(study_folder, threshold=threshold)

