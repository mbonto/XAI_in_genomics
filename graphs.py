import numpy as np
import scipy
from scipy.sparse import csc_matrix, eye, issparse
import networkx as nx
from networkx.algorithms import community


def get_a_graph(X, method='cosine_similarity'):
    """
    From a data matrix X, return an adjacency matrix A.
    
    Parameters:
      X  -- np.array of shape [# samples, # features]
    """
    if method == 'cosine_similarity':
        A = get_cosine_similarity(X)
    elif method == 'pearson_correlation':
        A = get_pearson_correlation(X)
    elif method == 'rank_correlation':
        A = get_rank_correlation(X)
    elif method == 'euclidean_distance':
        A = get_euclidean_distance(X)
    remove_diag(A)
    return A


def get_cosine_similarity(X):
    """
    Given a data matrix X, return the cosine similarity between all features.
    
    Parameters:
      X  -- np.array of shape [# samples, # features]
    """
    norm_X = np.linalg.norm(X, axis=0).reshape(1, -1)
    X = X / norm_X
    return np.matmul(X.T, X)


def get_pearson_correlation(X):
    """
    Given a data matrix X, return the Pearson correlation between all features.
    
    Parameters:
      X  -- np.array of shape [# samples, # features]
    """
    return np.corrcoef(X, rowvar=False)


def get_rank_correlation(X):
    """
    Given a data matrix X, return the Spearman correlation between all features.
    
    Parameters:
      X  -- np.array of shape [# samples, # features]
    """
    return scipy.stats.spearmanr(X, axis=0).correlation


def get_euclidean_distance(X):
    """
    Given a data matrix X, return the Euclidean distance between all features.
    
    Parameters:
      X  -- np.array of shape [# samples, # features]
    """
    out = (X.reshape(X.shape[0], 1, X.shape[1]) - X.reshape(X.shape[0], X.shape[1], 1))**2
    out = np.sum(out, axis=0)
    out = (np.max(out) - out) / np.max(out)
    return out


def remove_diag(A):
    """
    In-place modification of a matrix A setting its diagonal elements to 0.
    """
    np.fill_diagonal(A, np.zeros(A.shape[0]))

    
def make_a_graph_sparse(A, t_inf=None, t_sup=None):
    """
    Set to 0 99% of the off-diagonal coefficients of a symmetric square matrix A.
    """
    n_edge = A.shape[0] * (A.shape[0] - 1) / 2
    if t_inf is None:
        t_inf = int(n_edge / 100)
        print(f"We keep the weights of {t_inf} edges.")
        t_inf = -np.sort(-A[np.triu_indices(A.shape[0], k=1)])[t_inf]
    print(f"We keep the weights higher than {t_inf}.")
    A = (A >= t_inf) * A
    if t_sup is not None:
        A = (A <= t_sup) * A
        print(f"We keep the weights lower than {t_sup}.")
    return A


def create_consensus_matrix(G, n=10):
    """
    Parameters:
       G  --  nx graph
       n  --  int, number of partitions computed
    """
    D = np.zeros((nx.number_of_nodes(G), nx.number_of_nodes(G)))
    
    for _ in range(n):
        clusters = community.louvain.louvain_communities(G)
        for clst in clusters:
            D[np.ix_(list(clst), list(clst))] += np.ones((len(clst), len(clst)))
    D = D / n
    remove_diag(D)
    return D


def smoothness(G, signal):
    L = nx.laplacian_matrix(G)
    return np.matmul(np.matmul(signal, L.todense()), signal.T)


def get_degree_matrix(A):
    if issparse(A):
        return eye(A.shape[0], A.shape[1]).multiply(np.sum(A, axis=1))
    else:
        return np.diag(np.sum(A, axis=1))
    
    
def get_inverse_square_root_degree_matrix(A):
    if issparse(A):
        return eye(A.shape[0], A.shape[1]).multiply((1/np.sqrt(np.sum(A, axis=1))))
    else:
        return np.diag(1/np.sqrt(np.sum(A, axis=1)))


def get_normalized_adjaceny_matrix(A):
    if issparse(A):
        A = A + eye(A.shape[0], A.shape[1])
    else:
        A = A + np.eye(A.shape[0])
    D = get_inverse_square_root_degree_matrix(A)
    A = D.dot(A).dot(D)
    return A