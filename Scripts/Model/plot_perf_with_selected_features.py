# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import seaborn as sns
from sklearn.metrics import jaccard_score
from setting import *
from utils import *
from loader import *
from feature_selection import *
set_pyplot()


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
args = argParser.parse_args()
name = args.name
model_name = 'MLP'
exp = 1


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Load data
n_args = [1, 5, 10, 50, 100, 500, 1000]  # , 5000, 10000]

data = {}
for selection in ["var", "PCA_PC1", "F", "MI", "L1", "limma", "DESeq2", "IG"]:
    data[selection] = []
    for n in n_args:
        save_name = os.path.join(model_name, f"exp_{exp}_selection_{selection}_{n}")
        with open(os.path.join(save_path, save_name, "accuracy.csv"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(', ')
                if line[0] == 'train':
                    if float(line[1]) < 99:
                        print(f"{selection}_{n}_{float(line[1])}")
                if line[0] == 'balanced_test':
                    data[selection].append(float(line[1]))


# Save
label = {}
label["var"] = "Var"
label["PCA_PC1"] = "PCA" 
label["F"] = "F" 
label["MI"] = "MI" 
label["L1"] = "L1"
label["limma"] = "Limma"
label["DESeq2"] = "DESeq2"
label["IG"] = "IG" 


# Plot
plt.figure(figsize=(15, 5))
for selection in ["var", "PCA_PC1", "F", "MI", "L1", "limma", "DESeq2", "IG"]:
    plt.plot(n_args, data[selection], 'x-', label = label[selection])
    plt.xscale('log')
    plt.xlabel("Number of variables")
    plt.ylabel(f"Balanced accuracy")
    plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    plt.savefig(os.path.join(save_path, "figures", f"{model_name}_classif_plots.png"), bbox_inches='tight')

    
