# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import argparse
import json
from setting import *
from dataset import *


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name (BRCA, pancan, ttg-all, ttg-breast, BRCA-pam)")
args = argParser.parse_args()
name = args.name
print("Dataset", name)


# Paths
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Setting
database, cancer, label_name = get_TCGA_setting(name)


# Dataset
data = TCGA_dataset(data_path, database, cancer, label_name)


# Save
json.dump(data.genes_IDs, open(os.path.join(save_path, "genesIds.txt"),'w'))
