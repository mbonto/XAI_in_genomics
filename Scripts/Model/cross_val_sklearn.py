# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import argparse
from sklearn.model_selection import StratifiedKFold
from setting import *
from utils import *
from dataset import *
from loader import *
from plots_and_stats import *
from evaluate import *
from models import *
from training import *
from sklearn.linear_model import LogisticRegression


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-C", type=float, help="Inverse of regularization strength. Must be a positive float.", default=1)
args = argParser.parse_args()
name = args.name
C = args.C


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)
save_name = os.path.join("LR_L1_penalty", f"exp_C_{C}")


# Dataset
normalize = False
X, X_test, y, y_test, n_class, n_feat, class_name, feat_name = load_dataset(data_path, name, normalize)
print(f"In dataset for cross-validation, we have {n_class} classes and {X.shape[0]} examples. Each example contains {n_feat} features.")


# Model
clf = LogisticRegression(penalty='l1', C=C, max_iter=1000, solver='saga')


# Split function
n_split = 4
splits = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=0)


# Cross-validation
avg_train_balanced_score = 0
avg_val_balanced_score = 0

for fold, (train_idx, val_idx) in enumerate(splits.split(X, y)):
    print(f'\nFold {fold+1}')
    
    # Data
    cv_X = X[train_idx]
    cv_y = y[train_idx]
    cv_X_val = X[val_idx]
    cv_y_val = y[val_idx]
    
    # Transformation
    use_mean, use_std, log2, reverse_log2, divide_by_sum, factor = get_data_normalization_parameters(name)
    cv_X, cv_X_val = normalize_train_test_sets(cv_X, cv_X_val, use_mean, use_std, log2, reverse_log2, divide_by_sum, factor)

    
    # Train
    clf.fit(cv_X, cv_y)
    cv_y_pred = clf.predict(cv_X)
    train_score = accuracy_score(cv_y, cv_y_pred) * 100
    train_balanced_score = balanced_accuracy_score(cv_y, cv_y_pred) * 100

    # Test
    cv_y_val_pred = clf.predict(cv_X_val)
    val_score = accuracy_score(cv_y_val, cv_y_val_pred) * 100
    val_balanced_score = balanced_accuracy_score(cv_y_val, cv_y_val_pred) * 100
    
    # Store average scores
    avg_train_balanced_score += train_balanced_score
    avg_val_balanced_score += val_balanced_score
    
avg_train_balanced_score = avg_train_balanced_score / n_split
avg_val_balanced_score = avg_val_balanced_score / n_split

print('Final')
print(f'The balanced training accuracy with LR_L1_penalty is {np.round(avg_train_balanced_score, 2)}.')
print(f'The balanced test accuracy with LR_L1_penalty is {np.round(avg_val_balanced_score, 2)}.')


# Save
create_new_folder(os.path.join(save_path, save_name))
with open(os.path.join(save_path, save_name, "accuracy.csv"), "w") as f:
    f.write(f"balanced_train, {np.round(avg_train_balanced_score, 2)}\n")
    f.write(f"balanced_test, {np.round(avg_val_balanced_score, 2)}\n")
