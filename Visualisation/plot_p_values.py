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
args = argParser.parse_args()
name = args.name


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Load data
values_DE = np.load(os.path.join(save_path, "order", "order_DESeq2_values.npy"), allow_pickle=True)
values_DE = np.exp(- values_DE * np.log(10))
print("Minimal p-value: ", np.min(values_DE), "Maximal p-value: ", np.max(values_DE))


# Plot
save_name = os.path.join("figures", f"p_values_DESeq2_{name}.png")
plt.figure(figsize=(7, 2))
sns.displot(data=values_DE, kind="hist", color="blueviolet", binwidth=0.01, binrange=[0, 1])
plt.xlabel("Adjusted p-values")
plt.ylabel("Count")
plt.savefig(save_name, bbox_inches='tight', dpi=150)
# plt.show()

