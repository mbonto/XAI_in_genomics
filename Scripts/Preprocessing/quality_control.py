# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import json
import argparse
from setting import *
from download_data import *
from dataset import *
from plots_and_stats import *
from utils import *
from loader import *


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name (ttg-all, ttg-breast, BRCA-pam)")
args = argParser.parse_args()
name = args.name
print("Dataset", name)


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Unit of the gene expression levels
_, _, log2, reverse_log2, _, _ = get_data_normalization_parameters(name)

# Setting
database, cancer, label_name = get_TCGA_setting(name)


######################################### QC for samples ######################################### 
# The IDs of the samples whose expression level is null in more than 75% of the genes are written in ood_samples_{name}.npy.
if not os.path.isfile(os.path.join(data_path, database, "expression", f"ood_samples_{cancer}.npy")):
    # Dataset
    sample_IDs = TCGA_dataset(data_path, database, cancer, label_name).sample_IDs
    X, y, class_name, feat_name = load_data(data_path, name)
    
    # Transform data to the count space (instead of the log count space)
    if not log2:  # ttg-all, ttg-breast, BRCA-pam
        X = 2**X - 1
    elif log2 and reverse_log2:  # BRCA, KIRC
        X = 2**X - 1
    
    # Identify samples with more than t % of genes whose expression is 0
    t = 3 / 4
    n_zero_per_sample = np.sum(X == 0, axis=1)  # for each sample, number of genes whose expression level is 0
    samples_to_remove = np.argwhere(n_zero_per_sample > (X.shape[1] * t))[:, 0]
    print("Number of samples whose expression level is null in more than 75% of genes:", len(samples_to_remove))
    if len(samples_to_remove) != 0:
        class_list, n_gene = np.unique(y[samples_to_remove], return_counts=True)
        print("Number of samples per class")
        for c in range(len(class_list)):
            print(f"    {class_name[c]}: {n_gene[c]}")

    # Save the list of samples to remove
    samples_to_remove = list(np.array(sample_IDs)[samples_to_remove])
    np.save(os.path.join(data_path, database, "expression", f"ood_samples_{cancer}.npy"), samples_to_remove)
    del X


######################################### QC for genes ######################################### 
# [Using the training dataset only]
# The genes that have less than 5 counts in more than 99% samples are written in low_expressed_genes_{name}.npy.
# The genes that are constant are written in constant_genes_{name}.npy.

if not os.path.isfile(os.path.join(data_path, database, "expression", f"low_expressed_genes_{cancer}.npy")):
    # Dataset
    normalize = False
    X_train, X_test, y_train, y_test, n_class, n_feat, _, _ = load_dataset(data_path, name, normalize)

    # Transform data to the count space (instead of the log count space)
    if not log2:  # ttg-all, ttg-breast, BRCA-pam
        X_train = 2**X_train - 1
    elif log2 and reverse_log2:  # BRCA, KIRC
        X_train = 2**X_train - 1
    
    # Identify the genes that are weakly expressed
    t_gene = 5
    t_sample = int(X_train.shape[0] * 99 / 100)
    N = np.sum(X_train < t_gene, axis=0)  # samples where the expression is lower than t_gene
    genes_to_remove = (N > t_sample)  # genes with more than t_sample such samples
    print(f"Number of weakly expressed genes in more than {t_sample} samples: {np.sum(genes_to_remove)}")
    
    # Save the list of genes to remove
    genes_to_remove = np.argwhere(genes_to_remove)[:, 0]
    genes_to_remove = list(np.array(feat_name)[genes_to_remove])
    np.save(os.path.join(data_path, database, "expression", f"low_expressed_genes_{cancer}.npy"), genes_to_remove)

    # Identify the constant genes
    genes_to_remove2 = np.all(X_train == X_train[0, :], axis = 0)
    print("Number of constant genes:", np.sum(genes_to_remove2))
    genes_to_remove2 = np.argwhere(genes_to_remove2)[:, 0]
    genes_to_remove2 = list(np.array(feat_name)[genes_to_remove2])
    genes_to_remove2 = [gene for gene in genes_to_remove2 if gene not in genes_to_remove]
    print("Number of constant genes which are not weakly expressed:", len(genes_to_remove2))
    
    # Save the list of genes to remove
    np.save(os.path.join(data_path, database, "expression", f"constant_genes_{cancer}.npy"), genes_to_remove2)
