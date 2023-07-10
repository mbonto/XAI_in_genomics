import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE, spectral_embedding
from sklearn.cluster import KMeans, SpectralClustering, k_means
from sklearn.decomposition import DictionaryLearning
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import rand_score
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn


def do_PCA(X, n_components=None, save_name=None):
    # Parameter
    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])  ### Max number of eigenvectors
    # Train
    alg = PCA(n_components=n_components)
    alg.fit(X)
    # Control
    r = alg.explained_variance_ratio_
    plt.bar(x=np.arange(len(r))[:100], height=r[:100])
    plt.ylabel("Explained variance ratio\nby each PC")
    plt.xlabel("Index of the PC")
    if save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=150)
    plt.show()
    print(f"The total variance explained by the first {n_components} PC is {np.round(np.sum(r), 2)}.")
    return alg.transform(X), alg


def do_LDA(X, y, n_components=None):
    # Parameter
    if n_components is None:
        n_components = len(np.unique(y)) - 1  ### Max number of decision boundaries
    # Train
    alg = LDA(n_components=n_components)
    alg.fit(X, y)
    return alg.transform(X)


def do_KMeans(X, n_clusters=10):
    """
    Train KMeans on X of shape (n_sample, n_feat). Return the coordinates of the cluster centers in
    a matrix of shape (n_clusters, n_feat), the labels of each sample attributing them to a cluster, 
    the average Euclidean distance between each sample and the center of its cluster.
    """
    # Train
    alg = KMeans(n_clusters=n_clusters)
    alg.fit(X)
    return alg.cluster_centers_, alg.labels_, alg.inertia_


def learn_dict(X, n_components=None):
    """
    Learn a dictionary on X of shape (n_sample, n_feat). Return the trasnformed X in
    a matrix of shape (n_sample, n_atom) and the learned atoms in a matrix of shape (n_atom, n_feat).
    """    
    # Parameter
    if n_components is None:
        n_components = X.shape[1]  ### Number of features
    # Train
    alg = DictionaryLearning(n_components=n_components)
    alg.fit(X)
    return alg.transform(X), alg.components_


def do_TSNE(X, perplexity=30):
    """
    T-SNE is an algorithm used to visualize high dimensional data in lower 
    dimensions, so that points that are close in high dimensions remain 
    close in low dimensions. It does this in a non-linear and local way.
    
    Warning: Do not make assumptions on the relative sizes of the clusters 
    (with t-sne, dense clusters are expanded, spare clusters are skinked). 
    Warning: Do not interpret distances between well separated clusters.
    Warning: Clumps of points may just be noise, especially with low 
    perplexity values.
    
    Parameters:
        X  --  data of size (n_sample, n_feature)
        perplexity  --  parameter used as a proxy for the number of close 
                        neighbors each point has
    """
    alg = TSNE(n_components=2, perplexity=perplexity)
    return alg.fit_transform(X)


def choose_number_of_clusters_in_kmeans(X, n_clusters=np.arange(2, 10), random_state=1):
    inertia = []
    silhouette = []
    for n_cluster in n_clusters:
        print(f'# clusters {n_cluster}', end='\r')
        # K-means
        _, labels, inertia_ = k_means(
            X, n_cluster, random_state=random_state
        )
        # Scores
        inertia.append(inertia_)
        silhouette.append(silhouette_score(X, labels))
    # Plot
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(n_clusters, inertia, '+')
    plt.ylabel("Inertia")
    plt.subplot(2, 1, 2)
    plt.plot(n_clusters, silhouette, '+')
    plt.xlabel("Number of clusters")
    plt.ylabel("Average silhouette\n score")
    plt.show()
    
    
def choose_number_of_clusters_in_spectral_clustering(X, n_clusters=np.arange(2, 10), random_state=1):
    inertia = []
    silhouette = []
    for n_cluster in n_clusters:
        print(f'# clusters {n_cluster}', end='\r')
        # Affinity matrix
        model = SpectralClustering(n_clusters=n_cluster, random_state=random_state)
        model.fit(X)
        # Embedding
        maps = spectral_embedding(
            model.affinity_matrix_,
            n_components=n_cluster,
            random_state=random_state,
            drop_first=False)
        # K-means
        _, labels, inertia_ = k_means(
            maps, n_cluster, random_state=random_state
        )
        # Scores
        inertia.append(inertia_)
        silhouette.append(silhouette_score(maps, labels))
    # Plot
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(n_clusters, inertia, '+')
    plt.ylabel("Inertia")
    plt.subplot(2, 1, 2)
    plt.plot(n_clusters, silhouette, '+')
    plt.xlabel("Number of clusters")
    plt.ylabel("Average silhouette\n score")
    plt.show()
    
    

def kmeans_biclustering(X, n_clusters=[33, 2], n_iter=10):
    """
    Define an iterative biclustering composed of two steps. X is a np array of size (n_feat1, n_feat2).
    1/ Use Kmeans on X whose second dimension is averaged per cluster.
    
    2/ Use Kmeans on X whose first dimension is averaged per cluster.
    
    Return a plot showing the convergence of the algorithm in both dimensions.
    """
    labels1 = None
    labels2 = None
    conv1 = []
    conv2 = []
    for i in range(n_iter):
        conv1_, conv2_, labels1, labels2 = iterate_kmeans(X, labels1=labels1, labels2=labels2, n_clusters=n_clusters)
        conv1.append(conv1_)
        conv2.append(conv2_)
    # Plot
    plt.figure(figsize=(15, 3))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, n_iter), conv1[1:], '+')
    plt.xlabel("Number of iterations")
    plt.ylabel('Rand score\n between two iterations')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, n_iter), conv2[1:], '+')
    plt.xlabel("Number of iterations")
    plt.ylabel('Rand score\n between two iterations')
    plt.show()
    return labels1, labels2
    

        
def iterate_kmeans(X, labels1, labels2, n_clusters=[33, 2]):
    # Average the second dimension per cluster
    if labels2 is not None:
        X1 = np.zeros((X.shape[0], n_clusters[1]))
        for i in range(n_clusters[1]):
            X1[:, i] = np.mean(X[:, labels2 == i], axis=1)
    else:
        X1 = X.copy()
    
    # Compute K-means on X1
    alg1 = KMeans(n_clusters=n_clusters[0])
    alg1.fit(X1)
    
    # Measure the evolution of the clusters
    if labels1 is not None:
        conv1 = np.round(rand_score(labels1, alg1.labels_), 2)
    else:
        conv1 = 0
    labels1 = alg1.labels_
    
    # Average the first dimension per cluster
    X2 = np.zeros((n_clusters[0], X.shape[1]))
    for i in range(n_clusters[0]):
        X2[i, :] = np.mean(X[labels1 == i, :], axis=0)
        
    # Compute K-means on the second dimension
    alg2 = KMeans(n_clusters=n_clusters[1])
    alg2.fit(X2.T)
    
    # Measure the evolution of the clusters
    if labels2 is not None:
        conv2 = np.round(rand_score(labels2, alg2.labels_), 2)
    else:
        conv2 = 0
    labels2 = alg2.labels_
        
    return conv1, conv2, labels1, labels2


def cluster_nodes(dist_matrix, nodes, vmin=0, vmax=1, cut_dendogram=None, save_path=None):
    """
    Return a dictionary of clusters containing the indices of the attributed nodes. 
    Also, plot a clustermap and a dendogram to check the relevance of the created clusters.
    
    Parameters:
        dist_matrix  --  symmetric matrix of shape (d, d). dist_matrix[i, j] contains the distance between nodes i and j.
                         Here, distances should be normalized between 0 and 1 for better visualization.
        nodes  --  vector of shape (d, 1). nodes[i] contains the importance of node i for our application.
                   It is used for visualization purpose only.
        cut_dendogram  -- value used within the dendogram to define clusters
    """
    # Clustermap
    ## Continuous palette mapping the coefficents in area to colors
    cmap = sns.color_palette("mako_r", as_cmap=True)
    colors = cmap(nodes[:, 0])[:, :3]
    d = sns.clustermap(dist_matrix, col_colors=colors, row_colors=colors, cmap=cmap, vmin=vmin, vmax=vmax)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
    # Dendrogram
    den = scipy.cluster.hierarchy.dendrogram(d.dendrogram_col.linkage, color_threshold=cut_dendogram)
    plt.show()
    
    # Retrieve the clusters
    clusters = get_clusters_from_dendogram(den)
    return clusters, colors


def get_clusters_from_dendogram(d):
    """
    Return a dictionary of clustered leaves. E.g. {'C0': [0, 1], 'C1': [2]}.
    
    Parameter:
        d  --  dendogram obtained with scipy.cluster.hierarchy.dendrogram
    """
    clusters = np.unique(d['leaves_color_list'])
    _dict = {c: [] for c in clusters}
    for c, l in zip(d['leaves_color_list'], d['leaves']):
        _dict[c] += [l]
    return _dict

