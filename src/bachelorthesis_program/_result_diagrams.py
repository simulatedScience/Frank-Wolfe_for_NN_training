import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

from load_md_files import load_batch_info

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

def calculate_average(histories, metric: str = "val_loss"):
    total_metric = 0
    for history_obj in histories:
        total_metric += np.array(history_obj.history[metric])
    avg_loss = total_metric / len(histories)
    return avg_loss

def plot_metrics(ax, histories, batch_folder_path, metric: str = "val_loss"):
    color = next(ax._get_lines.prop_cycler)['color']
    avg_metric = calculate_average(histories, metric)
    n_epochs, label, _ = get_batch_info(batch_folder_path, metric)
    for history_obj in histories:
        epoch_wise_history = compress_history(n_epochs, history_obj.history[metric])
        ax.plot(epoch_wise_history, alpha=0.2, color=color)
    epoch_wise_avg = compress_history(n_epochs, avg_metric)
    ax.plot(epoch_wise_avg, linewidth=2, alpha=0.9, label=f'{label}', color=color)

def compress_history(n_epochs: int, batch_wise_history: np.array):
    """
    Simplify training history to include datapoints per epoch rather than per batch

    Args:
        n_epochs (int): number of epochs
        batch_wise_history (np.array): history of given metric with datapoints per batch

    Returns:
        (np.array): history of given metric with datapoints per epoch
    """
    n_batches = len(batch_wise_history)
        # average each epoch
    epoch_wise_history = np.zeros(n_epochs)
    for i in range(n_epochs):
        epoch_wise_history[i] = np.mean(
                batch_wise_history[i*n_batches//n_epochs:(i+1)*n_batches//n_epochs]
                )
    return epoch_wise_history

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

def plot_batches_group(folder_paths, title_addon: str = "", metric: str = "val_loss"):
    fig, ax = plt.subplots()
    # adjust subplot configuration to fit legend
    fig.subplots_adjust(right=0.975, left=0.075)
    # fig.subplots_adjust(right=0.85, left=0.1)
    for folder_path in folder_paths:
        histories = load_histories(folder_path)
        plot_metrics(
            ax=ax,
            histories=histories,
            batch_folder_path=folder_path,
            metric=metric)
    # ax.legend(bbox_to_anchor=(0.7, 0.55), loc='center left')
    ax.legend()
    plt.xlabel('Epoch number')
    plt.ylabel(f'{metric}')
    plt.title(f'{metric} evolution during training' + title_addon)
    plt.show()

def main(study_folder, metric: str = "val_loss"):
    history_groups = group_batches(study_folder)
    for hidden_layers, batch_folder_paths in history_groups.items():
        print(f"starting new group with hidden layers: {hidden_layers}")
        print(f"number of batches: {len(batch_folder_paths)}")
        plot_batches_group(batch_folder_paths, metric=metric, title_addon=f"\nhidden layers: {hidden_layers}")
        

if __name__ == '__main__':
    import tkfilebrowser
    # let user choose several folders to plot
    # folder_paths = tkfilebrowser.askopendirnames(title='Choose batch folders to plot')

    # # MNIST diagrams #1 (32, 64)
    # study_folder = "C:\\future_D\\uni\\Humboldt Uni\\Nebenhoerer SoSe 2023\\FW_NN_training\\src\\bachelorthesis_program\\training_info\\2023-08-07_10-43-21_mnist_reg_sfw_parameter_study"
    # main(study_folder, metric="val_accuracy")
    # main(study_folder, metric="accuracy")
    # main(study_folder, metric="val_loss")
    # main(study_folder, metric="loss")

    # # MNIST diagrams #2 (8, 16)
    # study_folder = "C:\\future_D\\uni\\Humboldt Uni\\Nebenhoerer SoSe 2023\\FW_NN_training\\src\\bachelorthesis_program\\training_info\\2023-08-07_12-43-04_mnist_reg_sfw_2_parameter_study"
    # main(study_folder, metric="val_accuracy")
    # main(study_folder, metric="accuracy")
    # main(study_folder, metric="val_loss")
    # main(study_folder, metric="loss")

    # # ChemReg diagrams #1 ((64, 16), (32, 32))
    # study_folder = "C:\\future_D\\uni\\Humboldt Uni\\Nebenhoerer SoSe 2023\\FW_NN_training\\src\\bachelorthesis_program\\training_info\\2023-08-07_15-30-46_chemreg_reg_sfw_parameter_study"
    # main(study_folder, metric="val_loss")
    # main(study_folder, metric="loss")

    # ChemReg diagrams #2
    study_folder = "C:\\future_D\\uni\\Humboldt Uni\\Nebenhoerer SoSe 2023\\FW_NN_training\\src\\bachelorthesis_program\\training_info\\2023-08-07_20-37-18__chemreg_reg_sfw_2_parameter_study"
    main(study_folder, metric="val_loss")
    main(study_folder, metric="loss")

    # # ChemReg paper diagrams
    # study_folder=r"C:\future_D\private\programming\python\AI experiments\hyperparameter_experiment\programs\neural net testing program 2_1\training_info\ML_paper_chemReg_cAdam_study"
    # main(study_folder, metric="val_loss")
    # main(study_folder, metric="loss")

