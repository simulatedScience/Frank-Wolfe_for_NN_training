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

def get_batch_info(batch_folder_path: str, metric: str = "accuracy"):
    """
    Simplify training history to include datapoints per epoch rather than per batch

    Args:
      history_obj: history object of keras model
    """
    # load batch info
    param_info, _ = load_batch_info(filename=os.path.join(batch_folder_path, "batch_parameters.md"))
    n_epochs = param_info["number_of_epochs"]
    hidden_layers = param_info["neurons_per_layer"]
    label = param_info["optimizer_params"]
    if label["optimizer_type"].lower() in ("adam", "sgd"):
        label = label["optimizer_type"]
        if "weight_decay" in label:
            label += f"+wd={label['weight_decay']}"
    elif label["constraints_type"].lower() in ("ksparse", "knorm"):
        label = f'{label["optimizer_type"]}+{label["constraints_type"]}+{label["constraints_K"]}'
    else:
        label = f'{label["optimizer_type"]}+{label["constraints_type"]}'
    print(f"Optimizer: {label}, hidden layers: {hidden_layers}")
    return n_epochs, label

def calculate_average(histories, metric: str = "accuracy"):
    total_metric = 0
    for history_obj in histories:
        total_metric += np.array(history_obj.history[metric])
    avg_loss = total_metric / len(histories)
    return avg_loss

def plot_metrics(ax, histories, batch_folder_path, metric: str = "accuracy"):
    color = next(ax._get_lines.prop_cycler)['color']
    avg_metric = calculate_average(histories, metric)
    n_epochs, label = get_batch_info(batch_folder_path, metric)
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

def main(folder_paths, metric: str = "accuracy"):
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
    plt.title(f'{metric} evolution during training')
    plt.show()

if __name__ == '__main__':
    import tkfilebrowser
    # let user choose several folders to plot
    folder_paths = tkfilebrowser.askopendirnames(title='Choose batch folders to plot')

    main(folder_paths)
