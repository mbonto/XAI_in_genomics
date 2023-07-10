# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
import argparse
from setting import *
from simulate_data import *
from loader import *
from graphs import *
from utils import *


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("--name", type=str, help="dataset name")
argParser.add_argument("--size", type=int, help="number of samples", default=9900)
args = argParser.parse_args()
name = args.name
n_sample = args.size
add_noise = False

# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)



# Simulation
print(f'         {name}           ')
alpha, eta, beta, proportion, n_gene, n_pathway, n_class, useful_paths, useful_genes = return_parameters(name)
X, y = generate_hierarchical_data(alpha, beta, n_sample, proportion)
if add_noise:
    X += np.random.uniform(0, 0.0001, X.shape)


# Save
dataset = {
    'X': X,
    'y': y,
    'n_class': n_class,
    'n_pathway': n_pathway,
    'n_gene': n_gene,
    'alpha': alpha,
    'eta': eta,
    'beta': beta,
    'useful_paths': useful_paths, 
    'useful_genes': useful_genes
}
np.save(os.path.join(data_path, name + '.npy'), dataset)



