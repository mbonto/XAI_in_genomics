# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import torch
from sklearn.metrics import balanced_accuracy_score
import argparse
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from setting import *
from utils import *
from dataset import *
from loader import *
from evaluate import *
from models import *
from training import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, GNN)")
argParser.add_argument("--layer", type=int, help="number of layers (MLP, GNN)", default=1)
argParser.add_argument("--feat", type=int, help="number of features per layer (MLP, GNN)", default=20)
argParser.add_argument("--graph_name", type=str, help="used only with a GCN model, name of the file containing the adjacency matrix of the graph")
args = argParser.parse_args()
name = args.name
model_name = args.model
n_layer = args.layer
n_hidden_feat = args.feat
graph_name = args.graph_name
print('Model    ', model_name)


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)
save_name = os.path.join(model_name, f"exp_layer_{n_layer}_feat_{n_hidden_feat}")


# Dataset
train_loader, test_loader, n_class, n_feat, class_name, feat_name, _, n_sample = load_dataloader(data_path, name, device)
print(f"In our dataset, we have {n_class} classes and {n_sample} examples. Each example contains {n_feat} features.")


# Preparation of the dataset
## Number of training samples
n_train_sample = 0
for sample, label  in train_loader:
    n_train_sample += len(label)
print(f"Number of training samples used for cross-validation: {n_train_sample}.")
## Data
X = torch.zeros((n_train_sample, n_feat), dtype=sample.dtype)
y = torch.zeros(n_train_sample, dtype=label.dtype)
count = 0
for sample, label in train_loader:
    batch_size = len(label)
    X[count:count + batch_size] = sample
    y[count:count + batch_size] = label
    count += batch_size
assert count == n_train_sample
dataset = custom_dataset(X, y)
## Split function
n_split = 4
splits = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=0)


# Cross-validation
avg_train_balanced_score = 0
avg_val_balanced_score = 0

for fold, (train_idx, val_idx) in enumerate(splits.split(X, y)):
    print(f'\nFold {fold+1}')
    
    # Data
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)
    
    # Transformation
    use_mean, use_std, log2, reverse_log2, divide_by_sum, factor = get_data_normalization_parameters(name)
    mean, std = find_mean_std(dataset, train_sampler, device, log2, reverse_log2, divide_by_sum, factor)
    if not use_mean:
        mean = torch.zeros(mean.shape).to(device)
    if not use_std:
        std = torch.ones(std.shape).to(device)
    transform = normalize(mean, std, log2, reverse_log2, divide_by_sum, factor)

    # Model
    softmax = False
    model = load_model(model_name, n_feat, n_class, softmax, device, save_path, n_layer, n_hidden_feat, graph_name)
    model.train()

    # Optimization
    criterion, optimizer, scheduler, n_epoch = set_optimizer(name, model)

    # Train
    epochs_acc = []
    epochs_loss = []
    for epoch in range(n_epoch):
        epoch_loss, epoch_acc = train(model, criterion, optimizer, train_loader, device, transform)  # train for 1 epoch
        print("\rLoss at epoch {}: {:.2f}. (Acc: {:.2f})".format(epoch+1, epoch_loss, epoch_acc*100), end='')
        scheduler.step()
    
    # Train score
    model.eval()
    y_pred, y_true = predict(model, train_loader, device, transform)
    train_score = compute_accuracy_from_predictions(y_pred, y_true)
    train_balanced_score = balanced_accuracy_score(y_true, y_pred) * 100

    # Validation score
    y_pred, y_true = predict(model, val_loader, device, transform)
    val_score = compute_accuracy_from_predictions(y_pred, y_true)
    val_balanced_score = balanced_accuracy_score(y_true, y_pred) * 100
    
    # Store average scores
    avg_train_balanced_score += train_balanced_score
    avg_val_balanced_score += val_balanced_score
    
avg_train_balanced_score = avg_train_balanced_score / n_split
avg_val_balanced_score = avg_val_balanced_score / n_split

print("")
print('Final')
print(f'The balanced training accuracy with our {model.name} is {np.round(avg_train_balanced_score, 2)}.')
print(f'The balanced test accuracy with our {model.name} is {np.round(avg_val_balanced_score, 2)}.')


# Save
create_new_folder(os.path.join(save_path, save_name))
with open(os.path.join(save_path, save_name, "accuracy.csv"), "w") as f:
    f.write(f"balanced_train, {np.round(avg_train_balanced_score, 2)}\n")
    f.write(f"balanced_test, {np.round(avg_val_balanced_score, 2)}\n")
