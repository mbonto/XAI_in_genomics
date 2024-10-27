# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from setting import *
from utils import *
from loader import *


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR_L1_penalty, LR_L2_penalty, xgboost)")
argParser.add_argument("-C", type=float, help="Inverse of regularization strength. Must be a positive float. Used by LR_L1_penalty, LR_L2_penalty.", default=1)
argParser.add_argument("--estimator", type=int, help="Number of trees used by xgboost", default=1)
argParser.add_argument("--max_depth", type=int, help="Maximal depth of each tree", default=1)
args = argParser.parse_args()
name = args.name
model_name = args.model
C = args.C
n_estimator = args.estimator
max_depth = args.max_depth


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)
save_name = os.path.join(model_name, f"exp_C_{C}") if model_name in ["LR_L1_penalty", "LR_L2_penalty"] else os.path.join(model_name, f"exp_estimator_{n_estimator}_depth_{max_depth}")


# Dataset
normalize = False  # Normalisation is applied after the training data has been split into a training set and a validation set.
X, X_test, y, y_test, n_class, n_feat, class_name, feat_name = load_dataset(data_path, name, normalize)
print(f"In dataset for cross-validation, we have {n_class} classes and {X.shape[0]} examples. Each example contains {n_feat} features.")


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

    # Model
    if model_name == "LR_L1_penalty":
        clf = LogisticRegression(penalty='l1', C=C, max_iter=1000, solver='saga')
    elif model_name == "LR_L2_penalty":
        clf = LogisticRegression(penalty='l2', C=C, max_iter=1000, solver='saga')
    elif model_name == "xgboost":
        clf = XGBClassifier(n_estimators=n_estimator, max_depth=max_depth, objective='binary:logistic') if n_class == 2 else XGBClassifier(n_estimators=n_estimator, max_depth=max_depth, objective='multi:softmax')


    # Train
    clf.fit(cv_X, cv_y)
    cv_y_pred = clf.predict(cv_X)
    train_score = accuracy_score(cv_y, cv_y_pred) * 100
    train_balanced_score = balanced_accuracy_score(cv_y, cv_y_pred) * 100

    # Validate
    cv_y_val_pred = clf.predict(cv_X_val)
    val_score = accuracy_score(cv_y_val, cv_y_val_pred) * 100
    val_balanced_score = balanced_accuracy_score(cv_y_val, cv_y_val_pred) * 100
    
    # Store average scores
    avg_train_balanced_score += train_balanced_score
    avg_val_balanced_score += val_balanced_score
    
avg_train_balanced_score = avg_train_balanced_score / n_split
avg_val_balanced_score = avg_val_balanced_score / n_split

print(f'Balanced training accuracy: {np.round(avg_train_balanced_score, 2)}. Balanced validation  accuracy: {np.round(avg_val_balanced_score, 2)}.')


# Save
create_new_folder(os.path.join(save_path, save_name))
with open(os.path.join(save_path, save_name, "accuracy.csv"), "w") as f:
    f.write(f"balanced_train, {np.round(avg_train_balanced_score, 2)}\n")
    f.write(f"balanced_val, {np.round(avg_val_balanced_score, 2)}\n")
