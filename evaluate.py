import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def count_correct_predictions(outputs, y):       
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == y).sum().item()
    return correct


def compute_accuracy_from_predictions(y_pred, y_true):
    """Count the number of items in y_pred and y_true that are equal at the same position and 
    return the count divided by the total number of elements in the lists. 
    
    Parameters:
        y_pred  -- Np array.
        y_true  -- Np array.
    """
    return np.round(np.mean(y_pred == y_true) * 100., 2)


def compute_accuracy_from_model(model, X, y):
    outputs = model(X)
    _, pred = torch.max(outputs.data, 1)
    return compute_accuracy_from_predictions(pred.detach().cpu().numpy(), y.detach().cpu().numpy().astype('int')[:, 0])


def compute_accuracy_from_model_with_dataloader(model, dataloader, transform, device):
    """Count the number of examples with a correct prediction and divide it by the number of examples.
    """
    acc = 0
    count = 0
    for x, target in dataloader:
        x = x.to(device)
        if transform:
            x = transform(x)
        target = target.to(device)
        y = torch.argmax(model(x), 1)
        acc += (torch.sum(y == target).detach().cpu().item())
        count += x.shape[0]
    return np.round(acc / count * 100, 2)


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval.
    
    Parameters:
        data -- Array containing an estimation of a value obtained across different data samples.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def get_confusion_matrix(y_true, y_pred, target_names, normalize='true'):
    cm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=target_names, normalize=normalize, values_format='.1f')
    plt.close('all')
    return cm


def plot_confusion_matrix(cm, file="draft", show=False):
    fig, ax = plt.subplots(figsize=(20, 20))
    cm.plot(ax=ax, values_format='.2f', colorbar=False)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(size=27)
    ax.set_xlabel('Predicted class', fontsize=22)
    ax.set_ylabel('True class', fontsize=22)
    for labels in cm.text_.ravel():
        labels.set_fontsize(12)
    fig.savefig(file, bbox_inches='tight')
    if show:
        plt.show()
    plt.close('all')
    
