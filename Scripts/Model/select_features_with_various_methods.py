import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
import argparse
from setting import *
from utils import *
from loader import *
from feature_selection import *


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
args = argParser.parse_args()
name = args.name


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


##############################################################################################
################################## Information on the datasets ###############################
##############################################################################################
# Data unit
## The unit of the gene expression is not the same across datasets.
## ttg-breast/all    unit: log2(norm_count + 1)  
## BRCA/KIRC         unit: log2(raw_count + 1)
## pancan            unit: norm_count
## BRCA-pam          unit: log2(norm_count + 1)


# Data normalization
## By default, different normalisation strategies are applied to the original data.
## ttg-breast/all    normalize: center and reduce the values associated to each gene.
## BRCA/KIRC         normalize: a) sum of the raw counts per sample is made equal to 10**6.
##                              b) center the values associated to each gene. 
## pancan            normalize: log2 transform + center and reduce the values associated to each gene.
## BRCA-pam          normalize: center and reduce the values associated to each gene.


# Normalization to apply for each method
## High variance / principal component / f test / mutual information
##     normalization: log2(norm_count + 1)  
##     Here, BRCA/KIRC: sum of counts per sample set to 10**6. pancan: log2 transform.
##     Note: for f test, the distribution of each gene has to look normal.
## DeSEQ2
##     normalization: norm_count
##     Ex: ttg-breast/all, BRCA/KIRC, BRCA-pam: 2**(data) - 1. BRCA/KIRC: sum of counts per sample set to 10**6.
## LR_L1_penalty/integrated_gradients
##     normalization: default strategy.


##############################################################################################
##################################### LOAD DATASETS ##########################################
##############################################################################################
X_train, X_test, y_train, y_test, n_class, n_feat, class_name, feat_name = load_dataset(data_path, name, normalize=False)
print(f"Number of classes: {n_class}.")
print(f"    Classes: {class_name}.")
print(f"Number of examples: {X.shape[0]}.")
print(f"Number of genes: {n_feat}.")
print(f"    Examples: {feat_name[:3]}.")
assert n_feat == len(np.unique(feat_name)) 
feat_name = np.array(feat_name)


##############################################################################################
####################### FEATURE SELECTION WITH LOG2(NORM_COUNT + 1) DATA #####################
##############################################################################################
# Prepare the data format: log2(norm_count + 1)
use_mean, use_std, log2, reverse_log2, divide_by_sum, factor = get_data_normalization_parameters(name)
use_mean = False
use_std = False
X, _ = normalize_train_test_sets(X_train, X_test, use_mean, use_std, log2, reverse_log2, divide_by_sum, factor)


# 1. Filter methods
print("High variance")
scores_var = select_features_with_high_variance(X, threshold=0.1)
print("Shape -", scores_var.shape, "Min -", np.round(np.min(scores_var), 2), "Max -",np.round(np.max(scores_var), 2))
print("Principal component analysis")
scores_PCA_PC1, _, _ = select_features_with_PCA(X, n_PC=3)
print("Shape -", scores_PCA_PC1.shape, "Min -", np.round(np.min(scores_PCA_PC1), 2), "Max -", np.round(np.max(scores_PCA_PC1), 2))
print("F test")
scores_F = select_features_with_F_test(X, y)
print("Shape -", scores_F.shape, "Min -", np.round(np.min(scores_F), 2), "Max -", np.round(np.max(scores_F), 2))
print("Mutual information")
scores_MI = select_features_with_mutual_information(X, y)
print("Shape -", scores_MI.shape, "Min -", np.round(np.min(scores_MI), 2), "Max -", np.round(np.max(scores_MI), 2))


##############################################################################################
########################### FEATURE SELECTION WITH NORM_COUNT DATA ###########################
##############################################################################################
# Prepare the data format: norm_count
# use_mean, use_std, log2, reverse_log2, divide_by_sum, factor = get_data_normalization_parameters(name)
# use_mean = False
# use_std = False
# if not log2:
#     reverse_log2 = True
# log2 = False
# X, _ = normalize_train_test_sets(X_train, X_test, use_mean, use_std, log2, reverse_log2, divide_by_sum, factor)


# Other filter methods
# print("DeSEQ2")


##############################################################################################
############################# FEATURE SELECTION WITH NORMALIZED DATA #########################
##############################################################################################
# Prepare the data format: default normalization
use_mean, use_std, log2, reverse_log2, divide_by_sum, factor = get_data_normalization_parameters(name)
X, X_test = normalize_train_test_sets(X_train, X_test, use_mean, use_std, log2, reverse_log2, divide_by_sum, factor)


# 2. Wrapper methods (IGNORED FOR NOW BC OF TIME CONSTRAINT)
# scores_RFE_10000 = select_features_with_RFE(X, y, 10000)    # time = > 474min to select 10000 features out of 58274 (disconnected from server before ending the run)
# scores_RFE_1000 = select_features_with_RFE(X, y, 1000)    # time = even longer
# scores_RFE_100 = select_features_with_RFE(X, y, 100)      # time = even looonger
# scores_RFE_10 = select_features_with_RFE(X, y, 10)        # time = even loooooonger
# scores_RFE_1 = select_features_with_RFE(X, y, 1)          # time = even loooooooonger
# scores_RFE = scores_RFE_10000 + scores_RFE_1000 + scores_RFE_100 + scores_RFE_10 + scores_RFE_1
# for test before running the proper RFE scores (select all features): ok
# scores_RFE = select_features_with_RFE(X, y, 58274)


# 3. Embedded methods
# print("LR with L1 penalty")
# C = get_hyperparameters(name, "LR_L1_penalty")
# scores_L1, _ = select_features_with_L1(X, y, X_test, y_test, save_path, C)
# if scores_L1.shape[0] == 1:
#     scores_L1 = scores_L1[0, :]
# else:
#     scores_L1 = np.mean(scores_L1, axis=0)
# print("Shape -", scores_L1.shape, "Min coef -", np.round(np.min(scores_L1), 2), "Max coef -", np.round(np.max(scores_L1), 2), "Number of selected features -", np.sum(scores_L1 != 0))

##############################################################################################
################################################## SAVE ######################################
##############################################################################################
order_var = np.argsort(-scores_var)  # genes sorted by decreasing variance
order_PCA_PC1 = np.argsort(-np.abs(scores_PCA_PC1))  # genes sorted by decreasing absolute importance on the first PC
order_F = np.argsort(-scores_F)  # genes sorted by decreasing F value
order_MI = np.argsort(-scores_MI)  # genes sorted by decreasing mutual information
# order_L1 = np.argsort(-np.abs(scores_L1))  # genes sorted by decreasing absolute value took by their associated parameter. A lot of zero coefficients

create_new_folder(os.path.join(save_path, "order"))
np.save(os.path.join(save_path, "order", "order_var.npy"), feat_name[order_var])
np.save(os.path.join(save_path, "order", "order_PCA_PC1.npy"), feat_name[order_PCA_PC1])
np.save(os.path.join(save_path, "order", "order_F.npy"), feat_name[order_F])
np.save(os.path.join(save_path, "order", "order_MI.npy"), feat_name[order_MI])
# np.save(os.path.join(save_path, "order", "order_L1.npy"), feat_name[order_L1])
