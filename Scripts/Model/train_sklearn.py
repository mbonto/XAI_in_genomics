# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import time
import argparse
from setting import *
from utils import *
from dataset import *
from loader import *
from plots_and_stats import *
from evaluate import *
from models import *
from training import *
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
set_pyplot()


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR_L1_penalty, KNN)")
argParser.add_argument("--exp", type=int, help="experiment number", default=1)
args = argParser.parse_args()
name = args.name
exp = args.exp
model_name = args.model
print(model_name)


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)
save_name = os.path.join(model_name, f"exp_{exp}")


# Dataset
normalize = True
X_train, X_test, y_train, y_test, n_class, n_feat, class_name, feat_name = load_dataset(data_path, name, normalize)
feat_name = np.array(feat_name)
print(f"In our dataset, we have {n_class} classes and {X_train.shape[0]} examples. Each example contains {n_feat} features.")


# Model
if model_name == "LR_L1_penalty":
    C = get_hyperparameters(name, model_name)
    clf = LogisticRegression(penalty='l1', C=C, max_iter=1000, solver='saga', random_state=exp)
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
print(f'The test accuracy is {np.round(test_score, 2)}.')
print(f'The balanced test accuracy is {np.round(test_balanced_score, 2)}.')


# Save
create_new_folder(os.path.join(save_path, save_name))
with open(os.path.join(save_path, save_name, "accuracy.csv"), "w") as f:
    f.write(f"train, {np.round(train_score, 2)}\n")
    f.write(f"balanced_train, {np.round(train_balanced_score, 2)}\n")
    f.write(f"test, {np.round(test_score, 2)}\n")
    f.write(f"balanced_test, {np.round(test_balanced_score, 2)}\n")
    f.write(f"duration, {np.round(duration, 2)}\n")
    if model_name == "LR_L1_penalty":
        f.write(f"# non-zero coefficients, {np.sum(np.mean(clf.coef_, axis=0) != 0)}\n")


if model_name == "LR_L1_penalty":
    np.save(os.path.join(save_path, save_name, "checkpoint.npy"), clf.coef_)


# Additionnal code for studying selected features
if model_name == "LR_L1_penalty":
    ## Shape of clf.coef_ is (1, n_feat) if n_class = 2 and (n_class, n_feat) otherwise.
    ## Here, the importance given to a feature is the absolute value of its coefficient (averaged over all classes when there are more than 2 classes). 
    if clf.coef_.shape[0] == 1:
        scores_L1 = clf.coef_[0, :]
    else:
        scores_L1 = np.mean(clf.coef_, axis=0)
    print("Shape -", scores_L1.shape, "Min coef -", np.round(np.min(scores_L1), 2), "Max coef -", np.round(np.max(scores_L1), 2), "Number of selected features -", np.sum(scores_L1 != 0))
    ## Features sorted by decreasing absolute values
    order_L1 = np.argsort(-np.abs(scores_L1))
    create_new_folder(os.path.join(save_path, "order"))
    np.save(os.path.join(save_path, "order", f"order_L1_exp_{exp}.npy"), feat_name[order_L1])


