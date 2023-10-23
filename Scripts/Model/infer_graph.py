# Libraries
import os
import sys
import numpy as np
from scipy.sparse import csc_matrix, eye, save_npz
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import argparse
from setting import *
from loader import *
from graphs import *
from utils import *


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("--method", type=str, default='pearson_correlation', help="method used to compute the adjacency matrix of the graph")
argParser.add_argument("-k", type=int, default=None, help="parameter used to find the threshold to apply on the adjacency matrix such that the graph contains n_node x k edges.")
argParser.add_argument("--min_value", type=float, default=None, help="parameter used to threshold the adjacency matrix of the graph")
args = argParser.parse_args()
name = args.name
method = args.method
k = args.k
min_value = args.min_value


# Path
data_path = get_data_path(name)
save_path = get_save_path(name, code_path)
create_new_folder(os.path.join(save_path, "graph"))


# Load dataset
X_train, _, y_train, _, _, _, _, _ = load_dataset(data_path, name, normalize=True)


# Infer the adjacency matrix between variables
print("Computing the adjacency matrix between variables...")
A = get_a_graph(X_train, method)
if not (A == A).all():
   print("If the variance of a variable is null, its correlation with other variables is not defined (nan value). Nan values are replaced here with 0.")
   nan_indices = np.argwhere(A != A)
   A[nan_indices[:, 0], nan_indices[:, 1]] = np.zeros(len(nan_indices))
   del nan_indices

# Sparse version
if k is not None:
    min_value = - np.sort(-A.reshape(-1))[A.shape[1] * k]

if min_value is not None:
    print(f"The threshold above which the edges are kept is {np.round(min_value, 2)}.")
    if len(A) <= 20000:
        A = (A >= min_value) * A
    else:  # slower but use less memory
        for i in range(len(A)):
            print(i, end='\r')
            A[i, :] = (A[i, :] >= min_value) * A[i, :] 
print('Correlation done')
save_npz(os.path.join(save_path, 'graph', f'{method}_{k}_variables'), csc_matrix(A))
with open(os.path.join(save_path, 'graph', f'{method}_{k}_variables_min_value.txt'), "w") as f:
    f.write(f"Minimal edge weight: {min_value}")

## Diffusion version
#print("Computing the diffusing matrix between variables...")
#if len(A) <= 20000:
#    A = (A > 0) * A
#else:  # slower but use less memory
#    for i in range(len(A)):
#        print(i, end='\r')
#        A[i, :] = (A[i, :] > 0) * A[i, :]
#A = csc_matrix(A)
#D = get_normalized_adjaceny_matrix(A)
#print('Diffusion done')
#save_npz(os.path.join(save_path, 'graph', f'{method}_{k}_variables_diffusion'), D)
#del D
del A


"""
# Infer the adjacency matrix between samples
print("Computing the adjacency matrix between samples...")
A = get_a_graph(X_train.T, method)

# Sparse version
if k is not None:
    min_value = - np.sort(-A.reshape(-1))[A.shape[0] * k]

if min_value is not None:
    print(f"The threshold above which the edges are kept is {np.round(min_value, 2)}.")
    if len(A) <= 20000:
        A = (A > min_value) * A
    else:  # slower but use less memory
        for i in range(len(A)):
            print(i, end='\r')
            A[i, :] = (A[i, :] > min_value) * A[i, :] 
save_npz(os.path.join(save_path, 'graph', f'{method}_{k}_samples'), csc_matrix(A))
with open(os.path.join(save_path, 'graph', f'{method}_{k}_samples_min_value.txt'), "w") as f:
    f.write(f"Minimal edge weight: {min_value}")
print('Correlation done')
"""
