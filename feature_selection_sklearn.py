# Librairies
import os
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2, VarianceThreshold, RFE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from utils import *


# Filters methods
## Without labels
def select_features_with_high_variance(X, threshold):
    """
    Return the variance of the features (shape (n_features)).
    VarianceThreshold selects all features whose variance is higher than `threshold`.
    """
    alg = VarianceThreshold(threshold=threshold)
    alg.fit(X)
    return alg.variances_


def select_features_with_PCA(X, n_PC=3):
    """
    Perform PCA.
    Print the explained variance of the first n_PC principal components.
    Return the scores of the features for the first 3 principal components.
    The shape of each principal component is (n_features).
    """
    pca = PCA(n_components=n_PC)
    out = pca.fit_transform(X)
    print(f"Number of principal components: {pca.components_.shape[0]}.")
    print(f"Explained variance of the first principal components:\n {pca.explained_variance_ratio_}.")
    return pca.components_[0,:], pca.components_[1,:], pca.components_[2,:]  ## pca.components is an array of shape (n_components, n_features)


## With labels
def select_features_with_F_test(X, y):
    """
    Return an array of shape (n_feat) containing the p-value associated with the F-statistic for each feature. 
    """
    scores, p = f_classif(X, y)  # p are the associated p_values.
    return p


def select_features_with_mutual_information(X, y):
    """
    Return an array of shape (n_feat) containing the mutual information between each feature X[:, i] and the labels y. 
    """
    return mutual_info_classif(X, y)


def select_features_with_chi2_test(X, y):
    """
    Return an array of shape (n_feat) containing the chi2-statistic for each feature. 
    """
    scores, p = chi2(X, y)
    return scores


# Wrapper methods
# def select_features_with_RFE(X, y, n_features):
#    svc = SVC(kernel="linear", C=1)
#    rfe = RFE(estimator=svc, n_features_to_select=n_features, step=1)
#    rfe.fit(X, y)
#    return rfe.ranking_
# Note 1: Use a different classifier.
# Note 2: RFE is really long to compute.
# Note3: Is the features ranking the same with different n_features to select (1 10 100 1000 10000)?


# Embedded methods
#def select_features_with_L1(X, y, X_test, y_test, save_path, C=1):
#    clf = LogisticRegression(penalty='l1', C=C, max_iter=1000, solver='saga')
#    # Train
#    clf.fit(X, y)
#    y_pred = clf.predict(X)
#    train_score = accuracy_score(y, y_pred) * 100
#    train_balanced_score = balanced_accuracy_score(y, y_pred) * 100
#    # Test
#    y_test_pred = clf.predict(X_test)
#    test_score = accuracy_score(y_test, y_test_pred) * 100
#    test_balanced_score = balanced_accuracy_score(y_test, y_test_pred) * 100
#    # Save
#    save_name = os.path.join("LR_L1_penalty", f"exp_feature_selection")
#    create_new_folder(os.path.join(save_path, save_name))
#    with open(os.path.join(save_path, save_name, "accuracy.csv"), "w") as f:
#        f.write(f"C, {C}\n")
#        f.write(f"train, {np.round(train_score, 2)}\n")
#        f.write(f"test, {np.round(test_score, 2)}\n")
#        f.write(f"balanced_train, {np.round(train_balanced_score, 2)}\n")
#        f.write(f"balanced_test, {np.round(test_balanced_score, 2)}\n")
#    return clf.coef_, clf.intercept_


def select_features_with_Tree(X, y):
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X, y)
    return forest.feature_importances_
