# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from joblib import dump
import time
import argparse
from setting import *
from utils import *
from loader import *
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR_L1_penalty, KNN, LR_L2_penalty)")
argParser.add_argument("--exp", type=int, help="experiment number", default=1)
argParser.add_argument("--selection", type=str, help="method used to select features (ex: MI, DESeq2, IG_LR_L1_penalty_set_train_exp_1, IG_MLP_set_train_exp_1)")
argParser.add_argument("--n_feat_selected", type=int, help="number of features selected.")
argParser.add_argument("--selection_type", type=str, choices=["best", "worst", "random_wo_best"], help="when `selection` is given, keep best, worst or random without best features.")
args = argParser.parse_args()
name = args.name
exp = args.exp
model_name = args.model
selection = args.selection
n_feat_selected = args.n_feat_selected
selection_type = args.selection_type
print('Model    ', model_name)


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)
save_name = os.path.join(model_name, f"exp_{exp}") if selection is None and n_feat_selected is None else os.path.join(model_name, f"FS")



# Seed
seed = exp if selection is None and n_feat_selected is None else exp + 100
random.seed(seed)
np.random.seed(seed)


# Dataset
normalize = True
studied_features = get_selected_features(selection, selection_type, n_feat_selected, save_path)
X_train, X_test, y_train, y_test, n_class, n_feat, class_name, feat_name = load_dataset(data_path, name, normalize, studied_features=studied_features)
if selection is None and n_feat_selected is None:
    print(f"In our dataset, we have {n_class} classes and {X_train.shape[0]} examples. Each example contains {n_feat} features.")


# Model
if model_name == "LR_L1_penalty":
    C = get_hyperparameters(name, model_name)
    clf = LogisticRegression(penalty='l1', C=C, max_iter=1000, solver='saga', random_state=seed)
elif model_name == "LR_L2_penalty":
    C = get_hyperparameters(name, model_name)
    clf = LogisticRegression(penalty='l2', C=C, max_iter=1000, solver='saga', random_state=seed)
elif model_name == "KNN":
    K = 3
    clf = KNeighborsClassifier(n_neighbors=K)


# Train
start_time = time.time()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
train_score = accuracy_score(y_train, y_pred) * 100
train_balanced_score = balanced_accuracy_score(y_train, y_pred) * 100
print(f'The training accuracy is {np.round(train_score, 2)}.')
duration = time.time() - start_time

# Test
y_pred = clf.predict(X_test)
test_score = accuracy_score(y_test, y_pred) * 100
test_balanced_score = balanced_accuracy_score(y_test, y_pred) * 100
if selection is None and n_feat_selected is None:
    print(f'The test accuracy is {np.round(test_score, 2)}.')
print(f'The balanced test accuracy is {np.round(test_balanced_score, 2)}.')


# Save
create_new_folder(os.path.join(save_path, save_name))
file_name = "accuracy.csv" if selection is None and n_feat_selected is None else f"{exp}_{selection}_{n_feat_selected}_{selection_type}.csv"
with open(os.path.join(save_path, save_name, file_name), "w") as f:
    f.write(f"train, {np.round(train_score, 2)}\n")
    f.write(f"balanced_train, {np.round(train_balanced_score, 2)}\n")
    f.write(f"test, {np.round(test_score, 2)}\n")
    f.write(f"balanced_test, {np.round(test_balanced_score, 2)}\n")
    f.write(f"duration, {np.round(duration, 2)}\n")
    if model_name == "LR_L1_penalty":
        f.write(f"# non-zero coefficients, {np.sum(np.mean(clf.coef_, axis=0) != 0)}\n")

if selection is None and n_feat_selected is None:
    dump(clf, os.path.join(save_path, save_name, "checkpoint.joblib"))






