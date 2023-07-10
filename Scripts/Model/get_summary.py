# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import argparse
import numpy as np
from setting import *

# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="Dataset name")
argParser.add_argument("-m", "--model", type=str, help="Model name (LR, MLP, DiffuseLR, DiffuseMLP)")
argParser.add_argument("--n_repet", type=int, help="Results are averaged for all experiments between 1 and `n_repet`")
args = argParser.parse_args()
name = args.name
model_name = args.model
n_repet = args.n_repet
print('Model    ', model_name)

# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)

# Summarize results
exps = np.arange(1, n_repet+1)
test_acc = []
train_acc = []

for exp in exps:
    save_name = os.path.join(model_name, f"exp_{exp}")
    
    with open(os.path.join(save_path, save_name, "accuracy.csv"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(', ')
            if line[0] == 'train':
                assert float(line[1]) > 99
                train_acc.append(float(line[1]))
            if line[0] == 'balanced_test':
                test_acc.append(float(line[1]))

assert len(test_acc) == len(exps)
print(f"Train accuracy with {model_name} on {name}: {np.round(np.mean(train_acc) , 2)} +- {np.round(np.std(train_acc) , 2)}")   
print(f"Balanced test accuracy with {model_name} on {name}: {np.round(np.mean(test_acc) , 2)} +- {np.round(np.std(test_acc) , 2)}")   
