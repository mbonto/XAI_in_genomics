###############################################################################################################
################################################ LIBRARIES ####################################################
###############################################################################################################
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
sys.path.append('./XAI_in_progress/')
print(sys.path)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2, VarianceThreshold, RFE
from sklearn.metrics import jaccard_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from setting import *
from utils import *
from loader import *

set_pyplot()

###############################################################################################################
######################################### FUNCTION DEFINITION #################################################
###############################################################################################################
name = 'BRCA'  
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)

# Univariate feature selection
## Without labels
def select_features_with_high_variance(X, threshold):
    """
    Select all features whose variance is higher some threshold.
    """
    alg = VarianceThreshold(threshold=threshold)
    alg.fit(X)
    return alg.variances_

# PCA
def select_features_with_PCA(X, n_PC):
    """
    Perform PCA
    Print the explained variance of the first n_PC principal components
    Return the scores of the features for the first 3 principal components
    """
    pca = PCA(n_components=n_PC)
    out = pca.fit_transform(X)

    print(f"Number of principal components: {pca.components_.shape[0]}.")
    print(f"Explained variance of the first principal components:\n {pca.explained_variance_ratio_}.")
    
    return pca.components_[0,:], pca.components_[1,:], pca.components_[2,:]

# Univariate feature selection
## With labels
def select_features_with_F_test(X, y):
    scores, p = f_classif(X, y)
    return scores

def select_features_with_mutual_information(X, y): 
    return mutual_info_classif(X, y)

def select_features_with_chi2_test(X, y):
    scores, p = chi2(X, y)
    return scores

# RFE - More useful but reaaaaaally Longer to compute
def select_features_with_RFE(X, y, n_features):
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=n_features, step=1)
    rfe.fit(X, y)
    return rfe.ranking_

# If time to compute the results, try to compute it for different n_features to select (1 10 100 1000 10000)
# And sum all the obtained scores to get a final unique RFE score 
# with the objectif to compare the sets with other methods like in Myriam set intersection methods
# see below Feature selection methods

# L1-based - Lasso
def select_features_with_L1(X, y):
    alpha = 1
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
    lasso.fit(X, y)
    return np.abs(lasso.coef_) # The importance of a feature is the absolute value of its coefficient

# not very informative - only two features have a coef value different from 0

# Tree-based 
def select_features_with_Tree(X, y):
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X, y)
    return forest.feature_importances_

# Normalisation of the data
# We assume that the unit of the gene expression is log2(raw count +1).
## If normalize is True, several normalisations are applied to the data.
## 1. The sum of the raw counts per sample is made equal to 10**6.
## 2. The average value of each gene (computed on the training set) is removed from each gene. 
normalize = False

###############################################################################################################
############################################# DATA LOADING ####################################################
###############################################################################################################

# Load BRCA dataset
X, X_test, y, y_test, n_class, n_feat, class_name, feat_name = load_dataset(data_path, name, normalize)

# For PyTorch only: use load_dataloader to train/test a neural network.
# train_loader, test_loader, n_class, n_feat, class_name, feat_name, transform, n_sample = load_dataloader(data_path, name, device)
# print(f"In our dataset, we have {n_class} classes and {n_sample} examples. Each example contains {n_feat} features.")

print(f"Number of classes: {n_class}.")
print(f"    Classes: {class_name}.")
print(f"Number of genes: {X.shape[1]}.")
print(f"Examples of names: {feat_name[:3]}.")
print(f"Number of training examples: {X.shape[0]}.")
print(f"Number of test examples: {X_test.shape[0]}.")

###############################################################################################################
########################################## FEATURE SELECTION ##################################################
###############################################################################################################
# Parameters
threshold = 0.1  # for "select_features_with_high_variance" only
n_PC = 10 # for "select_features_with_PCA" the number of first PCs for which the explained variance is printed
n_features = 1 # for select_features_with_RFE, the number of features to select

# Features scores with each feature selection method
# 1. Dimension reduction
scores_var = select_features_with_high_variance(X, threshold)
scores_PCA_PC1, scores_PCA_PC2, scores_PCA_PC3 = select_features_with_PCA(X, 10)

# 2. Filters
scores_F = select_features_with_F_test(X, y)
scores_MI = select_features_with_mutual_information(X, y)
scores_chi2 = select_features_with_chi2_test(X, y)

# 3. Wrapper / embedded (IGNORED FOR NOW BC OF TIME CONSTRAINT)
# If time to compute the RFE results, try to compute it for different n_features to select (1 10 100 1000 10000)
# And sum all the obtained scores to get a final unique RFE score 
# with the objectif to compare the sets with other methods like in Myriam set intersection methods
# see:
# scores_RFE_10000 = select_features_with_RFE(X, y, 10000)    # time = > 474min to select 10000 features out of 58274 (disconnected from server before ending the run)
# scores_RFE_1000 = select_features_with_RFE(X, y, 1000)    # time = even longer
# scores_RFE_100 = select_features_with_RFE(X, y, 100)      # time = even looonger
# scores_RFE_10 = select_features_with_RFE(X, y, 10)        # time = even loooooonger
# scores_RFE_1 = select_features_with_RFE(X, y, 1)          # time = even loooooooonger
# scores_RFE = scores_RFE_10000 + scores_RFE_1000 + scores_RFE_100 + scores_RFE_10 + scores_RFE_1
# for test before running the proper RFE scores (select all features): ok
# scores_RFE = select_features_with_RFE(X, y, 58274)

scores_L1 = select_features_with_L1(X, y)
scores_Tree = select_features_with_Tree(X, y)

# Load the features ranked by IG for all BRCA samples
model_name = "MLP"  # "MLP" or "LR"
set_name = "train"
exp = 1  # exp between 1 and 10

scores_IG = np.load(sys.path.append(f"./Gdc/Data/scores/{model_name}_{exp}_{set_name}_avg.npy"))

###############################################################################################################
############################################ METHOD OVERLAP ###################################################
###############################################################################################################

order_IG = np.argsort(-scores_IG)
order_var = np.argsort(-scores_var)
order_PCA = np.argsort(-scores_PCA_PC1)
order_F = np.argsort(-scores_F)
order_MI = np.argsort(-scores_MI)
order_chi2 = np.argsort(-scores_chi2)
order_L1 = np.argsort(-scores_L1)
order_Tree = np.argsort(-scores_Tree)

n_args = [1, 10, 100, 1000, 10000]
avg_jaccard_scores_var = []
avg_jaccard_scores_PCA = []
avg_jaccard_scores_F = []
avg_jaccard_scores_MI = []
avg_jaccard_scores_chi2 = []
avg_jaccard_scores_L1 = []
avg_jaccard_scores_Tree = []

for n in n_args:
    set_IG = set(order_IG[:n])
    set_var = set(order_var[:n])
    set_PCA = set(order_PCA[:n])
    set_F = set(order_F[:n])
    set_MI = set(order_MI[:n])
    set_chi2 = set(order_chi2[:n])
    set_L1 = set(order_L1[:n])
    set_Tree = set(order_Tree[:n])

    avg_jaccard_scores_var.append(len(list(set_IG.intersection(set_var))) / n)
    avg_jaccard_scores_PCA.append(len(list(set_IG.intersection(set_PCA))) / n)
    avg_jaccard_scores_F.append(len(list(set_IG.intersection(set_F))) / n)
    avg_jaccard_scores_MI.append(len(list(set_IG.intersection(set_MI))) / n)
    avg_jaccard_scores_chi2.append(len(list(set_IG.intersection(set_chi2))) / n)
    avg_jaccard_scores_L1.append(len(list(set_IG.intersection(set_L1))) / n)
    avg_jaccard_scores_Tree.append(len(list(set_IG.intersection(set_Tree))) / n)

###############################################################################################################
############################################## SAVE + PLOT ####################################################
###############################################################################################################
# Save the jaccard scores 
jaccard_all_scores = {'var': avg_jaccard_scores_var, 
                      'PCA': avg_jaccard_scores_PCA, 
                      'F': avg_jaccard_scores_F, 
                      'MI': avg_jaccard_scores_MI, 
                      'chi2': avg_jaccard_scores_chi2,
                      'L1': avg_jaccard_scores_L1,
                      'Tree': avg_jaccard_scores_Tree}

np.save(os.path.join("jaccard_all_scores", f"{model_name}_{exp}_{set_name}.npy"), jaccard_all_scores)

# Show the Jaccard scores
plt.figure(figsize=(15, 5))
plt.plot(n_args, avg_jaccard_scores_var, 'x-', label = "vs Var")
plt.plot(n_args, avg_jaccard_scores_PCA, 'x-', label = "vs PCA")
plt.plot(n_args, avg_jaccard_scores_F, 'x-', label = "vs F")
plt.plot(n_args, avg_jaccard_scores_MI, 'x-', label = "vs MI")
plt.plot(n_args, avg_jaccard_scores_chi2, 'x-', label = "vs chi2")
plt.plot(n_args, avg_jaccard_scores_L1, 'x-', label = "vs L1")
plt.plot(n_args, avg_jaccard_scores_Tree, 'x-', label = "vs Tree")
plt.xlabel("Number of variables")
plt.ylabel(f"Average Jaccard score \n(IG vs FS methods)")
plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
plt.savefig(os.path.join("jaccard_plots", f"{model_name}_{exp}_{set_name}_avg.png"))
