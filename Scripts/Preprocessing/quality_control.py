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
argParser.add_argument("-n", "--name", type=str, help="dataset name (BRCA, pancan, ttg-all, ttg-breast, BRCA-pam)")
args = argParser.parse_args()
name = args.name
print("Dataset", name)


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Setting
database, cancer, label_name = get_TCGA_setting(name)


######################################### QC for samples ######################################### 
# The IDs of the samples whose expression level is null in more than 75% of the genes are written in ood_samples_{name}.npy.
# Dataset
weakly_expressed_genes_removed = False
ood_samples_removed = False
normalize_expression = False
sample_IDs = TCGA_dataset(data_path, database, cancer, label_name, weakly_expressed_genes_removed, ood_samples_removed, normalize_expression).sample_IDs
X, y, class_name, feat_name = load_data(data_path, name, weakly_expressed_genes_removed, ood_samples_removed, normalize_expression)

# Transform data to the count space (instead of the log count space)
if database in ['gdc', 'ttg', 'legacy']:  # ttg-all, ttg-breast, BRCA-pam, BRCA, KIRC
    X = 2**X - 1
print(f"Minimal gene expression value in the whole dataset (should be around 0): {np.round(np.min(X), 2)}.")


# Identify samples with more than t % of genes whose expression is 0
t = 3 / 4
n_zero_per_sample = np.sum(X <= 0, axis=1)  # for each sample, number of genes whose expression level is <= 0
samples_to_remove = np.argwhere(n_zero_per_sample > (X.shape[1] * t))[:, 0]
print(f"Number of samples whose expression level is null in more than 75% of genes: {np.round(len(samples_to_remove), 2)}.")
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
# The genes that have less than 5 counts in more than 75% samples in all classes are written in low_expressed_genes_{name}.npy.
## The genes that are constant are written in constant_genes_{name}.npy.

# Dataset
## normalize: remove mean and divide by std each gene
## normalize_expression: make the sum of the gene expression of each sample equal to 10^6 and return log2(norm_count + 1)
X_train, X_test, y_train, y_test, n_class, n_feat, _, _ = load_dataset(data_path, name, 
                                                                       normalize=False, 
                                                                       weakly_expressed_genes_removed=False, 
                                                                       ood_samples_removed=True, normalize_expression=True)

# Normalize data
X_train = 2**X_train - 1
print(f"Minimal gene expression value in the training set (should be around 0): {np.min(X_train)}.") 


# Identify the genes that are weakly expressed in all classes
t_gene = 5
per_sample = 75 / 100
genes_to_remove = set(np.arange(n_feat))
print(f"Genes with less than {t_gene} counts in {per_sample * 100}% of the training samples of each class:")
for y in range(n_class):
    N = np.sum(X_train[y_train==y] < t_gene, axis=0)  # samples where the expression is lower than t_gene
    t_sample = int(X_train[y_train==y].shape[0] * per_sample)
    temp_set = np.argwhere((N > t_sample))  # genes with more than t_sample such samples
    print(f"    class {y} ({X_train[y_train==y].shape[0]} samples): {temp_set.shape[0]}")
    temp_set = temp_set[:, 0]
    genes_to_remove = genes_to_remove.intersection(set(temp_set))  # genes with more than t_sample such samples
genes_to_remove = list(genes_to_remove)
print(f"Number of weakly expressed genes: {len(genes_to_remove)}.")

# Save the list of genes to remove
genes_to_remove = list(np.array(feat_name)[genes_to_remove])
np.save(os.path.join(data_path, database, "expression", f"low_expressed_genes_{cancer}.npy"), genes_to_remove)

# Identify the constant genes
cst_genes = np.all(X_train == X_train[0, :], axis = 0)
# print("Number of constant genes:", np.sum(cst_genes))
cst_genes = np.argwhere(cst_genes)[:, 0]
cst_genes = list(np.array(feat_name)[cst_genes])
cst_genes = [gene for gene in cst_genes if gene not in genes_to_remove]
assert len(cst_genes) == 0, "Be careful: some genes are constant. Dividing them with their standard deviation will lead to NAN values."

# Save the list of genes to remove
# np.save(os.path.join(data_path, database, "expression", f"constant_genes_{cancer}.npy"), genes_to_remove2)
