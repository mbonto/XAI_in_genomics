# Librairies
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.decomposition import PCA


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
    return pca.components_[0,:], pca.components_[1,:], pca.components_[2,:]  # pca.components is an array of shape (n_components, n_features)


def select_features_with_mutual_information(X, y):
    """
    Return an array of shape (n_feat) containing the mutual information between each feature X[:, i] and the labels y. 
    """
    return mutual_info_classif(X, y)



