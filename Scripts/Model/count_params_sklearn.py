# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
from joblib import load
import argparse


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR_L1_penalty, KNN)")
argParser.add_argument("--exp", type=int, help="experiment number", default=1)
args = argParser.parse_args()
name = args.name
model_name = args.model
exp = args.exp


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)
save_name = os.path.join(model_name, f"exp_{exp}")


# Model
if model_name == "LR_L1_penalty":
    clf = load(os.path.join(save_path, save_name, "checkpoint.joblib"))
coef_param = clf.coef_.shape[0] * clf.coef_.shape[1]
intercept_param = clf.intercept_.shape[0]
print("Number of parameters", coef_param + intercept_param)


