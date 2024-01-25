# Librairies
import os
import sys
code_path = os.path.split(os.getcwd())[0]
sys.path.append(code_path)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from setting import *
from utils import *
set_pyplot()


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--method", type=str, help="method name")
args = argParser.parse_args()
name = args.name
method = args.method


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Load data
values = np.load(os.path.join(save_path, "order", f"order_{method}_values.npy"), allow_pickle=True)
values = np.exp(- values * np.log(10))
print("Minimal adjusted p-value: ", np.min(values), "Maximal adjusted p-value: ", np.max(values), "# < 0.05: ", np.sum(values < 0.05))


# Plot
save_name = os.path.join("figures", f"adjusted_p_values_{method}_{name}.png")
plt.figure(figsize=(7, 2))
sns.displot(data=values, kind="hist", color="blueviolet", binwidth=0.01, binrange=[0, 1])
plt.xlabel("Adjusted p-values")
plt.ylabel("Count")
plt.yscale("log")
plt.savefig(save_name, bbox_inches='tight', dpi=150)
# plt.show()

