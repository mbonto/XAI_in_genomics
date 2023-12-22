import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
import argparse
from setting import *
from utils import *
from loader import *
from feature_selection_sklearn import *


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
## ttg-breast/all    unit: log2(count_uq + 1)  
## BRCA/KIRC         unit: log2(count + 1)
## pancan            unit: count_uq
## BRCA-pam          unit: log2(count_uq + 1)


# Data normalisation.
## By default, the same normalisation strategy is applied to all datasets.
##                   unit after normalisation: log2(norm_count + 1)

# Data normalisation used for feature selection methods.
## High variance / principal component / f test / mutual information
##                   unit after normalisation: log2(norm_count + 1)  
## Note: for f test, the distribution of each gene has to look normal.
## DeSEQ2
##                  unit after normalisation: norm_count
## LR_L1_penalty, MLP, LR, GCN
##                   unit after normalisation: log2(norm_count + 1) + each gene is centered and reduced 


##############################################################################################
##################################### LOAD DATASETS ##########################################
##############################################################################################
X_train, X_test, y_train, y_test, n_class, n_feat, class_name, feat_name = load_dataset(data_path, name, normalize=False)
print(f"Number of classes: {n_class}.")
print(f"    Classes: {class_name}.")
print(f"Number of examples: {X_train.shape[0]}.")
print(f"Number of genes: {n_feat}.")
print(f"    Examples: {feat_name[:3]}.")
assert n_feat == len(np.unique(feat_name)) 
feat_name = np.array(feat_name)


##############################################################################################
####################### FEATURE SELECTION WITH LOG2(NORM_COUNT + 1) UNIT #####################
##############################################################################################
# 1. Filter methods
print("High variance")
X = X_train.copy()
scores_var = select_features_with_high_variance(X, threshold=0.1)
print("Shape -", scores_var.shape, "Min -", np.round(np.min(scores_var), 2), "Max -",np.round(np.max(scores_var), 2))
print("Principal component analysis")
X = X_train.copy()
scores_PCA_PC1, _, _ = select_features_with_PCA(X, n_PC=3)
print("Shape -", scores_PCA_PC1.shape, "Min -", np.round(np.min(scores_PCA_PC1), 2), "Max -", np.round(np.max(scores_PCA_PC1), 2))
# print("F test")
# X = X_train.copy()
# y = y_train.copy()
# scores_F = select_features_with_F_test(X, y)
# scores_F = -np.log10(scores_F)
# print("Shape -", scores_F.shape, "Min -", np.round(np.min(scores_F), 10), "Max -", np.round(np.max(scores_F), 2))
print("Mutual information")
X = X_train.copy()
y = y_train.copy()
scores_MI = select_features_with_mutual_information(X, y)
print("Shape -", scores_MI.shape, "Min -", np.round(np.min(scores_MI), 2), "Max -", np.round(np.max(scores_MI), 2))


##############################################################################################
############################# FEATURE SELECTION WITH NORMALIZED DATA #########################
##############################################################################################
# Prepare the data format: default normalization
# use_mean, use_std, log2, reverse_log2, divide_by_sum, factor = get_data_normalization_parameters(name)
# X, X_test = normalize_train_test_sets(X_train, X_test, use_mean, use_std, log2, reverse_log2, divide_by_sum, factor)


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
# scores_L1, _ = select_features_with_L1(X, y_train, X_test, y_test, save_path, C)
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
# order_F = np.argsort(-scores_F)  # genes sorted by increasing log10(p-value) associated with the F statistic
order_MI = np.argsort(-scores_MI)  # genes sorted by decreasing mutual information


create_new_folder(os.path.join(save_path, "order"))
np.save(os.path.join(save_path, "order", "order_var.npy"), feat_name[order_var])
np.save(os.path.join(save_path, "order", "order_var_values.npy"), scores_var[order_var])
np.save(os.path.join(save_path, "order", "order_PCA_PC1.npy"), feat_name[order_PCA_PC1])
np.save(os.path.join(save_path, "order", "order_PCA_PC1_values.npy"), scores_PCA_PC1[order_PCA_PC1])
# np.save(os.path.join(save_path, "order", "order_F.npy"), feat_name[order_F])
# np.save(os.path.join(save_path, "order", "order_F_values.npy"), scores_F[order_F])
np.save(os.path.join(save_path, "order", "order_MI.npy"), feat_name[order_MI])
np.save(os.path.join(save_path, "order", "order_MI_values.npy"), scores_MI[order_MI])

