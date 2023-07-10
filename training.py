import torch
import torch.nn as nn
import numpy as np
from evaluate import *


def train(model, criterion, optimizer, data_loader, device, transform=None, shuffle_feat=None, class_map=None):
    """ Train a neural network for one epoch.
    """
    model.train()
    epoch_loss = 0.
    epoch_count = 0.
    epoch_total = 0.
    for i, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)
        # Normalize input data
        if transform:
            x = transform(x)
        # Shuffle features
        if shuffle_feat is not None:
            x = x[:, shuffle_feat]
        # Reattribute classes
        if class_map is not None:
            y = torch.tensor(list(map(lambda x: class_map[x.item()], y))).to(device)
        # Zero the parameter gradients.
        optimizer.zero_grad()
        # Forward + backward + optimize.
        outputs = model(x)
        # print(outputs)
        loss = criterion(outputs, y)
        loss.backward()  
        optimizer.step()
        count = count_correct_predictions(outputs.clone().detach(), y)
        # Statistics.
        epoch_loss += loss.item()
        epoch_count += count
        epoch_total += y.size(0)
    return epoch_loss / (i+1), epoch_count / epoch_total
    
    
def predict(model, loader, device, transform=None, shuffle_feat=None, class_map=None):
    """Return the predictions of a model over the examples contained in loader."""
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            # Normalize input data
            if transform:
                x = transform(x)
            # Shuffle features
            if shuffle_feat is not None:
                x = x[:, shuffle_feat]
            # Reattribute classes
            if class_map is not None:
                y = torch.tensor(list(map(lambda x: class_map[x.item()], y))).to(device)
            # Forward.
            outputs = model(x)
            _, pred = torch.max(outputs.data, 1)
            y_true = y_true + list(y.cpu().detach().numpy())
            y_pred = y_pred + list(pred.cpu().detach().numpy())
    return np.array(y_pred), np.array(y_true)