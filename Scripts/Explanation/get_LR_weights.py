# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
import random
import torch
from joblib import load
from sklearn.metrics import accuracy_score
import argparse
from setting import *
from utils import *
from loader import *
from evaluate import *
from models import *
from XAI_method import *
set_pyplot()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, LR_L1_penalty, LR_L2_penalty)")
argParser.add_argument("--exp", type=int, help="experiment number", default=1)
args = argParser.parse_args()
name = args.name
model_name = args.model
exp = args.exp
print('Model    ', model_name)


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)
save_name = os.path.join(model_name, f"exp_{exp}")
create_new_folder(os.path.join(save_path, "order"))


# Seed
random.seed(exp)
np.random.seed(exp)
torch.manual_seed(exp)
if device == 'cuda':
    torch.cuda.manual_seed_all(exp)


# Dataset
train_loader, test_loader, n_class, n_feat, class_name, feat_name, transform, n_sample = load_dataloader(data_path, name, device)
feat_name = np.array(feat_name)


# Model
if model_name == "LR":
    softmax = True
    n_layer, n_hidden_feat, graph_name = get_hyperparameters(name, model_name)
    model = load_model(model_name, n_feat, n_class, softmax, device, save_path, n_layer, n_hidden_feat, graph_name)
    checkpoint = torch.load(os.path.join(save_path, save_name, 'checkpoint.pt'))
    # print(checkpoint['state_dict'])
    # print(checkpoint["state_dict"]['fc.weight'].dtype, checkpoint["state_dict"]['fc.weight'].shape)
    # print(checkpoint["state_dict"]['fc.bias'].dtype, checkpoint["state_dict"]['fc.bias'].shape)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
else:
    model = load(os.path.join(save_path, save_name, "checkpoint.joblib"))
    checkpoint = {}
    with open(os.path.join(save_path, save_name, "accuracy.csv"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(', ')
            if line[0] == 'train':
                checkpoint["train_acc"] = float(line[1])
            elif line[0] == 'test':
                checkpoint["test_acc"] = float(line[1])


# Assert that the model and the data are coherent
if model_name == "LR":
    assert compute_accuracy_from_model_with_dataloader(model, train_loader, transform, device) == checkpoint['train_acc']
    assert compute_accuracy_from_model_with_dataloader(model, test_loader, transform, device) == checkpoint['test_acc']
else:
    correct = 0.
    total = 0
    for X, y in train_loader:
        X = X.to(device)
        if transform:
            X = transform(X)
        X = X.detach().cpu().numpy()
        correct += np.sum(y.detach().cpu().numpy() == model.predict(X))
        total += X.shape[0]
    assert np.round(correct / total * 100, 2) == checkpoint['train_acc']
    correct = 0.
    total = 0
    for X, y in test_loader:
        X = X.to(device)
        if transform:
            X = transform(X)
        X = X.detach().cpu().numpy()
        correct += np.sum(y.detach().cpu().numpy() == model.predict(X))
        total += X.shape[0]
    assert np.round(correct / total * 100, 2) == checkpoint['test_acc']


# Parameters of the logistic regression
if model_name == "LR":
    params = model.fc.weight  # shape (n_class, n_feat) with more than 2 classes or (1, n_feat)
    params = params.detach().cpu().numpy()
else:
    params = model.coef_  # shape (n_class, n_feat) with more than 2 classes or (1, n_feat)


# Save the features sorted by decreasing absolute value of their parameters (averaged over all classes when n_class > 2)
if n_class == 2:
    assert params.shape[0] == 1
    scores = np.abs(params).reshape(-1)
else:
    scores = np.mean(np.abs(params), axis=0).reshape(-1)
print("Shape -", scores.shape, "Min coef -", np.round(np.min(scores), 2), "Max coef -", np.round(np.max(scores), 2), "Number of selected features -", np.sum(scores != 0))
order = np.argsort(-scores)
np.save(os.path.join(save_path, "order", f"order_weight_{model_name}_exp_{exp}.npy"), feat_name[order])
np.save(os.path.join(save_path, "order", f"order_weight_{model_name}_exp_{exp}_values.npy"), scores[order])



