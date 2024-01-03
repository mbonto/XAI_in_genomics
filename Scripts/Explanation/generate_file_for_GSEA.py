# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
import argparse
from setting import *
from utils import *


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, GCN, LR_L1_penalty, DESeq2, MI)")
argParser.add_argument("--exp", type=int, help="experiment number")
args = argParser.parse_args()
name = args.name
model_name = args.model
exp = args.exp


# Path
save_path = get_save_path(name, code_path)


# Load the list of genes ranked by order of importance for the model
set_name = {"LR": f"order_IG_LR_set_train_exp_{exp}",
            "MLP": f"order_IG_MLP_set_train_exp_{exp}",
            "GCN": f"order_IG_GCN_set_train_exp_{exp}",
            "LR_L1_penalty": f"order_IG_LR_L1_penalty_set_train_exp_{exp}",
            "DESeq2": "order_DESeq2",
            "MI": "order_MI"}
gene_list = np.load(os.path.join(save_path, "order", set_name[model_name] + ".npy"), allow_pickle=True)
gene_values = np.load(os.path.join(save_path, "order", set_name[model_name] + "_values.npy"), allow_pickle=True)


# Save the list in a txt file
# Genes should be represented by their gene symbol.
# For BRCA, gene names need to be converted from ENSG.
if name == "BRCA":
    convert_name = {}
    with open(os.path.join(save_path, "order", "ENSG_to_symbols.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            ENSG = line.strip().split("\t")[0]
            if len(line.strip().split("\t")) != 2:
                convert_name[ENSG] = "unknown"
                # print(f"{ENSG}")
            else:
                symbol = line.strip().split("\t")[1]
                convert_name[ENSG] = symbol


create_new_folder(os.path.join(save_path, "GSEA"))
with open(os.path.join(save_path, "GSEA", set_name[model_name] + ".txt"), "w") as f:
    for g, gene in enumerate(gene_list):
        if name == "pancan":
            f.write(f"{gene.split('|')[0]} - {gene_values[g]}\n")
        elif name == "BRCA":
            f.write(f"{convert_name[gene]} - {gene_values[g]}\n")
        else:
            f.write(f"{gene} - {gene_values[g]}\n")


with open(os.path.join(save_path, "GSEA", set_name[model_name] + "_wo_values.txt"), "w") as f:
    for g, gene in enumerate(gene_list):
        if name == "pancan":
            f.write(f"{gene.split('|')[0]}\n")
        elif name == "BRCA":
            f.write(f"{convert_name[gene]}\n")
        else:
            f.write(f"{gene}\n")





