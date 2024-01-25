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
argParser.add_argument("--set", type=str, help="set (train or test)")
argParser.add_argument("--exp", type=int, help="experiment number", default=1)
args = argParser.parse_args()
name = args.name
model_name = args.model
set_name = args.set
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


# Set
if set_name == 'train':
    loader = train_loader
elif set_name == 'test':
    loader = test_loader
    
    
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


"""
if n_class == 2:
    # Compute the feature effects for each sample
    effect = np.zeros((n_sample, n_feat))
    y_pred = np.ones((n_sample), dtype='int')
    y_true = np.ones((n_sample), dtype='int')
    count = 0
    for x, target in loader:
        batch_size = x.shape[0]
        x = x.to(device)
        if transform:
            x = transform(x)
    
        # Prediction
        if model_name == "LR":
            outputs = model(x)
            pred = ((outputs.data > 0.5).reshape(-1) * 1).detach().cpu().numpy()
        else:
            pred = model.predict(x.detach().cpu().numpy())

        y_true[count:count + batch_size] = target.cpu().detach().numpy()
        y_pred[count:count + batch_size] = pred
    
        # Effect
        effect[count:count + batch_size, :] = x.detach().cpu().numpy() * params  # (batch_size, n_feat)
    
        count = count + batch_size
    

    # Keep only correctly classified examples
    correct_indices = np.argwhere((y_pred - y_true) == 0)[:, 0]
    print("There are {} uncorrect examples. We remove the uncorrect examples from our study.".format(len(y_pred) - len(correct_indices)))
    effect = effect[correct_indices]
    y_true = y_true[correct_indices]
    y_pred = y_pred[correct_indices]
    
    
    # Compute the feature effects averaged per class
    effect_per_class = np.zeros((n_class, n_feat))
    count = np.zeros((n_class, 1))
    for i in range(effect.shape[0]):
        effect_per_class[y_true[i]] += effect[i]
        count[y_true[i], 0] += 1
    effect_per_class = effect_per_class / count
    print("Effect per class", effect_per_class.shape, "min", np.round(np.min(effect_per_class), 2), "max", np.round(np.max(effect_per_class), 2))
    
    
    # Compute the feature effects averaged over all studied features
    base_class, studied_class = get_XAI_hyperparameters(name, n_class)
    avg_effect = np.zeros((n_feat))
    for c in studied_class:
        avg_effect += effect_per_class[c]
    avg_effect = avg_effect / len(studied_class)
    print("Effect averaged over studied classes", avg_effect.shape, "min", np.round(np.min(avg_effect), 2), "max", np.round(np.max(avg_effect), 2))
    
    
    # Compute the feature effects with respect to the baseline
    baseline = get_baseline(train_loader, device, n_feat, transform, base_class)  # shape (1, n_feat)
    baseline = baseline.detach().cpu().numpy()
    baseline_effect = (baseline * params).reshape(-1)
    avg_effect_wrt_baseline = avg_effect - baseline_effect
    print("Effect for the baseline", baseline_effect.shape, "min", np.round(np.min(baseline_effect), 2), "max", np.round(np.max(baseline_effect), 2))
    print("Averaged effect - baseline effect", avg_effect_wrt_baseline.shape, "min", np.round(np.min(avg_effect_wrt_baseline), 2), "max", np.round(np.max(avg_effect_wrt_baseline), 2))
    
    
    # Save
    order = np.argsort(-avg_effect)
    np.save(os.path.join(save_path, "order", f"order_effect_{model_name}_exp_{exp}.npy"), feat_name[order])
    np.save(os.path.join(save_path, "order", f"order_effect_{model_name}_exp_{exp}_values.npy"), scores[order])
    order = np.argsort(-avg_effect_wrt_baseline)
    np.save(os.path.join(save_path, "order", f"order_effect_wrt_baseline_{model_name}_exp_{exp}.npy"), feat_name[order])
    np.save(os.path.join(save_path, "order", f"order_effect_wrt_baseline_{model_name}_exp_{exp}_values.npy"), scores[order])

"""


