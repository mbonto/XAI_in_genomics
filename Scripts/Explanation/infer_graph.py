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
from XAI_method import *


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, DiffuseLR, DiffuseMLP)")
argParser.add_argument("--exp", type=int, help="experiment number", default=1)
argParser.add_argument("--set", type=str, help="set (train or test)")
argParser.add_argument("--method", type=str, default='pearson_correlation', help="method used to compute the adjacency matrix of the graph")
argParser.add_argument("--min_value", type=float, default=None, help="minimal value appearing in the adjacency matrix of the graph")
args = argParser.parse_args()
name = args.name
model_name = args.model
exp = args.exp
set_name = args.set
method = args.method
min_value = args.min_value
XAI_method = "Integrated_Gradients"  # "Expected_Gradients", "Integrated_Gradients"

# Path
data_path = get_data_path(name)
save_path = get_save_path(name, code_path)
save_path = os.path.join(save_path, model_name, f"exp_{exp}", XAI_method)
create_new_folder(os.path.join(save_path, "graph"))


# Load explanations
X, y_pred, y, _, _, _, _ = load_attributions(XAI_method, save_path, set_name)
# Only correctly classified examples are considered
X = X[np.argwhere(y_pred == y).reshape(-1)]
y = y[np.argwhere(y_pred == y).reshape(-1)]


# Infer the adjacency matrix between variables
print("Computing the correlation matrix between variables...")
A = get_a_graph(X, method)
if not (A == A).all():
   print("If the variance of a variable is null, its correlation with other variables is not defined (nan value). Nan values are replaced here with 0.")
   nan_indices = np.argwhere(A != A)
   A[nan_indices[:, 0], nan_indices[:, 1]] = np.zeros(len(nan_indices))
   del nan_indices

# Sparse version
if min_value is not None:
    if len(A) <= 20000:
        A = (np.abs(A) > min_value) * A
    else:  # slower but use less memory
        for i in range(len(A)):
            print(i, end='\r')
            A[i, :] = (np.abs(A[i, :]) > min_value) * A[i, :]
print('Correlation done')
save_npz(os.path.join(save_path, "graph", f'{set_name}_{method}_{min_value}_variables'), csc_matrix(A))

# Diffusion version
print("Computing the diffusing matrix between variables...")
if min_value is not None:
    if len(A) <= 20000:
        A = (A > 0) * A
    else:  # slower but use less memory
        for i in range(len(A)):
            print(i, end='\r')
            A[i, :] = (A[i, :] > 0) * A[i, :]
A = csc_matrix(A)
D = get_normalized_adjaceny_matrix(A)
save_npz(os.path.join(save_path, "graph", f'{set_name}_{method}_{min_value}_variables_diffusion'), D)
del A, D
print("Diffusion done")


# Infer the adjacency matrix between samples
print("Computing the correlation matrix between samples...")
A = get_a_graph(X.T, method)

# Sparse version
if min_value is not None:
    if len(A) <= 20000:
        A = (np.abs(A) > min_value) * A
    else:  # slower but use less memory
        for i in range(len(A)):
            print(i, end='\r')
            A[i, :] = (np.abs(A[i, :]) > min_value) * A[i, :]
save_npz(os.path.join(save_path, "graph", f'{set_name}_{method}_{min_value}_samples'), csc_matrix(A))
