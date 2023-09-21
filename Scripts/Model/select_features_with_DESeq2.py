# Code adapted from the appyter-catalog github repository (https://github.com/MaayanLab/appyter-catalog/tree/main),
# distributed under an Attribution-NonCommercial-ShareAlike 4.0 International license.

# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
from setting import *
from dataset import *
from utils import *
from loader import *
from feature_selection_r import * 
import pandas as pd
import argparse
import numpy as np


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
args = argParser.parse_args()
name = args.name
method = "DESeq2" # "limma" or "DESeq2"


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Loading data
X, X_test, y, y_test, n_class, n_feat, class_name, feat_name = load_dataset(data_path, name, normalize=False)
print(f"Number of classes: {n_class}.")
print(f"    Classes: {class_name}.")
print(f"Number of examples: {X.shape[0]}.")
print(f"Number of genes: {n_feat}.")
print(f"    Examples: {feat_name[:3]}.")
feat_name = np.array(feat_name)
# database = 'ttg'
# label_name = '_sample_type'
# data = TCGA_dataset(data_path, database, name, label_name)


# Prepare data
## Limma and Deseq2 require un-normalized counts or estimated counts of sequencing reads.
## The unit of the gene expression levels is not the same across datasets.
###    ttg-breast/all    unit: log2(norm_count + 1)  
###    BRCA/KIRC         unit: log2(raw_count + 1)
###    pancan            unit: norm_count
###    BRCA-pam          unit: log2(norm_count + 1)
## When applicable, we have to inverse the log2-transform: 2**data - 1.
## For BRCA and KIRC, we also have to normalize the raw counts to make them comparable across samples: sum of counts per sample set to 10**6. 
use_mean, use_std, log2, reverse_log2, divide_by_sum, factor = get_data_normalization_parameters(name)
use_mean = False
use_std = False
if not log2:
    reverse_log2 = True
log2 = False
X, _ = normalize_train_test_sets(X, X_test, use_mean, use_std, log2, reverse_log2, divide_by_sum, factor)


# Convert data into dataframe (pandas)
## Expression dataset. Rows = genes. Columns = samples. Values = gene expression level.
# expression1 = data.expression.T
# expression1 = expression1.apply(lambda x: 2**x - 1)
expression = pd.DataFrame(data=X.T, index=feat_name, columns=['sample_'+str(i) for i in range(X.shape[0])])
# print("Min >= 0, Max", expression.max().max(), expression.min().max())
## Phenotype dataset. Rows = samples. Columns = category.
# phenotype = data.labels
assert n_class == 2
y = np.where(y == 0, class_name[0], class_name[1])
phenotype = pd.DataFrame(data=y.reshape(-1, 1), index=['sample_'+str(i) for i in range(X.shape[0])], columns=['_sample_type'])
phenotype = phenotype['_sample_type']
# print(phenotype.head())


# Method
assert n_class == 2
base_class, studied_class = get_XAI_hyperparameters(name, n_class)
classes = []
classes.append(class_name[studied_class[0]])
classes.append(class_name[base_class])
print("Condition vs Control", classes)
signatures = get_signatures(classes, expression, phenotype, method)


# Plot
for label, signature in signatures.items():
    print("Table containing the gene expression signature generated from a differential gene expression analysis {} (Condition vs Control).".format(label))
    print(signature.head())
    
pvalue_threshold = 0.05
logfc_threshold = 1.5
plot_type = 'static'

results = {}
for label, signature in signatures.items():
    results[label] = run_volcano(signature, label, pvalue_threshold, logfc_threshold, plot_type)
    print("Volcano plot for {}. Interactive scatter plot which displays the log2-fold changes and statistical significance of each gene calculated by performing a differential gene expression analysis. Genes with logFC > {} and p-value < {} in red and genes with logFC < -{} and p-value < {} in blue. Additional information for each gene is available by hovering over it.".format(label, logfc_threshold, pvalue_threshold, logfc_threshold, pvalue_threshold))
    plot_volcano(results[label], os.path.join(save_path, "figures"))

    
# Save
create_new_folder(os.path.join(save_path, "order"))
for label, signature in signatures.items():
    scores = results[label]['x']
    feat_name = scores.index
    order = np.argsort(-scores)
    order = feat_name[order]
    np.save(os.path.join(save_path, "order", f"order_{method}.npy"), order)


