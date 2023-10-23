# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
from scipy.sparse import load_npz
import networkx as nx
import argparse
from setting import *


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("--graph_name", type=str, help="used only with a GCN model, name of the file containing the adjacency matrix of the graph")
args = argParser.parse_args()
name = args.name
graph_name = args.graph_name


# Path
data_path = get_data_path(name)
save_path = get_save_path(name, code_path)


# Load the graph
A = load_npz(os.path.join(save_path, "graph", graph_name))
G = nx.from_scipy_sparse_matrix(A)


# Show statistics
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of connected components: {nx.number_connected_components(G)}")
print("Connected components")
size_cc = {}
for c in nx.connected_components(G):
    if len(c) not in size_cc.keys():
        size_cc[len(c)] = 0
    size_cc[len(c)] += 1
for s in sorted(size_cc.keys()):
    print(f"    Size {s}: {size_cc[s]}")

