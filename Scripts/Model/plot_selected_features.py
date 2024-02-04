# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import seaborn as sns
from sklearn.metrics import jaccard_score
from setting import *
from utils import *
from loader import *
set_pyplot()


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
args = argParser.parse_args()
name = args.name
n_repet = 10


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Load data
feat_name = np.array(json.load(open(os.path.join(save_path, "genesIds.txt"))))
n_feat = len(feat_name)
order_VAR = {1: np.load(os.path.join(save_path, "order", "order_var.npy"))}
order_PCA = {1: np.load(os.path.join(save_path, "order", "order_PCA_PC1.npy"))}
order_MI = {1: np.load(os.path.join(save_path, "order", "order_MI.npy"))}
order_DE = {1: np.load(os.path.join(save_path, "order", "order_DESeq2.npy"), allow_pickle=True)} 
order_R = {1: np.load(os.path.join(save_path, "order", "order_edgeR.npy"), allow_pickle=True)} 
order_weight_L1 = {}
order_weight_L2 = {}
order_weight_LR = {}
order_IG_MLP = {}
order_IG_LR = {}
order_IG_GCN = {}
order_IG_L1 = {}
order_IG_L2 = {}
for exp in range(1, n_repet+1):
    order_weight_L1[exp] = np.load(os.path.join(save_path, "order", f"order_weight_LR_L1_penalty_exp_{exp}.npy"))
    order_weight_L2[exp] = np.load(os.path.join(save_path, "order", f"order_weight_LR_L2_penalty_exp_{exp}.npy"))
    order_weight_LR[exp] = np.load(os.path.join(save_path, "order", f"order_weight_LR_exp_{exp}.npy"))
    order_IG_MLP[exp] = np.load(os.path.join(save_path, "order", f"order_IG_MLP_set_train_exp_{exp}.npy"), allow_pickle=True) 
    order_IG_LR[exp] = np.load(os.path.join(save_path, "order", f"order_IG_LR_set_train_exp_{exp}.npy"), allow_pickle=True) 
    order_IG_GCN[exp] = np.load(os.path.join(save_path, "order", f"order_IG_GCN_set_train_exp_{exp}.npy"), allow_pickle=True) 
    order_IG_L1[exp] = np.load(os.path.join(save_path, "order", f"order_IG_LR_L1_penalty_set_train_exp_{exp}.npy"))
    order_IG_L2[exp] = np.load(os.path.join(save_path, "order", f"order_IG_LR_L2_penalty_set_train_exp_{exp}.npy"))


assert len(order_VAR[1]) == n_feat
assert len(order_PCA[1]) == n_feat
assert len(order_MI[1]) == n_feat
assert len(order_DE[1]) == n_feat
assert len(order_R[1]) == n_feat
for exp in range(1, n_repet+1):
    assert len(order_weight_L1[exp]) == n_feat
    assert len(order_weight_L2[exp]) == n_feat
    assert len(order_weight_LR[exp]) == n_feat
    assert len(order_IG_MLP[exp]) == n_feat
    assert len(order_IG_LR[exp]) == n_feat
    assert len(order_IG_GCN[exp]) == n_feat
    assert len(order_IG_L1[exp]) == n_feat
    assert len(order_IG_L2[exp]) == n_feat



# Heatmaps
n_args = [10, 100] # 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
hm = {}

for n in n_args:
    data = {
            'VAR': {},
            'PCA': {},
            'MI': {},
            'DESeq2': {},
            'EdgeR': {},
            'LR+L1 (weight)': {},
            'LR+L2 (weight)':{},
            'LR+L1 (IG)': {},
            'LR+L2 (IG)': {},
            'MLP': {},
            'GNN':{}
            }
    data['VAR'][1] = list(order_VAR[1][:n])
    data['PCA'][1] = list(order_PCA[1][:n])
    data['MI'][1] = list(order_MI[1][:n])
    data['DESeq2'][1] =  list(order_DE[1][:n])
    data['EdgeR'][1] =  list(order_R[1][:n])
    for exp in range(1, n_repet+1):
        data['LR+L1 (weight)'][exp] = list(order_weight_L1[exp][:n])
        data['LR+L2 (weight)'][exp] = list(order_weight_L2[exp][:n])
        data['LR+L1 (IG)'][exp] = list(order_IG_L1[exp][:n])
        data['LR+L2 (IG)'][exp] = list(order_IG_L2[exp][:n])
        data['MLP'][exp] = list(order_IG_MLP[exp][:n])
        data['GNN'][exp] = list(order_IG_GCN[exp][:n])
    x = []
    for k1 in data.keys():
        for k2 in data.keys():
            n_repet_k1 = len(data[k1].keys())
            n_repet_k2 = len(data[k2].keys())
            count = 0
            for i in range(1, n_repet_k1+1):
                for j in range(1, n_repet_k2+1):
                    if k1 == k2 and i == j:
                        pass
                    else:
                        count += len(set(data[k1][i]) & set(data[k2][j]))
            if k1 == k2 and n_repet_k1 == 1:
                count = len(set(data[k1][i]))
            else:
                count = count / (n_repet_k1 * n_repet_k2) if k1 != k2 else count / (n_repet_k1 * n_repet_k2 - n_repet_k1)
            x.append((k1, k2, count))

    df = pd.DataFrame(x).pivot(index=0, columns=1, values=2)
    df = df[['VAR', 'PCA', 'MI', 'EdgeR', 'DESeq2', 'LR+L1 (weight)', 'LR+L1 (IG)', 'LR+L2 (weight)', 'LR+L2 (IG)', 'MLP', 'GNN']]
    df = df.reindex(['VAR', 'PCA', 'MI', 'EdgeR', 'DESeq2', 'LR+L1 (weight)', 'LR+L1 (IG)', 'LR+L2 (weight)', 'LR+L2 (IG)', 'MLP', 'GNN'])
    if n == 10:
        df_10 = (df.copy() * 10).round().astype(int)
        matrix_triu = np.triu(np.ones(np.shape(df_10)))
    elif n == 100:
        df_100 = (df.copy()).round().astype(int)
        matrix_tril = np.tril(np.ones(np.shape(df_100)), -1)
    plt.figure(figsize=(20, 20))
    sns.heatmap(df)
    plt.xlabel('')
    plt.ylabel('')
    # plt.savefig(os.path.join(save_path, "figures", f"heatmap_FS_{n}.png"), bbox_inches='tight')


# Heatmap 10 vs 100 features
plt.figure(figsize=(20, 20))
sns.heatmap(df_10, mask=matrix_triu, vmin=0, vmax=100, cbar=False, annot=True, cmap="BuPu", annot_kws={"fontsize":30}, fmt='d', square=True)
h = sns.heatmap(df_100, mask=matrix_tril, vmin=0, vmax=100, annot=True, cmap="BuPu", annot_kws={"fontsize":30}, cbar_kws={"ticks":[0, 100], "shrink": .81}, fmt='d', square=True)
# h.tick_params(bottom=False, labelbottom=False)  # to remove bottom ticks and tick labels
# h.tick_params(left=False, labelleft=False)  # to remove bottom ticks and tick labels
# Labels
plt.xlabel('Upper triangular: comparison among 100 features', loc='right', labelpad=50, fontsize=30)
# plt.xlabel('', loc='right', labelpad=50, fontsize=30)
plt.ylabel('Lower triangular: comparison among 10 features', loc='bottom', labelpad=50, fontsize=30)
# plt.ylabel('', loc='bottom', labelpad=50, fontsize=30)
ax = plt.gca()
ax.xaxis.set_label_position('top')
# Ticks
ax.tick_params(axis='both', which='major', labelsize=30)
# Colorbar
cbar = h.figure.axes[-1]
cbar.tick_params(labelsize=30)
cbar.set_yticklabels(['0%', '100%'])
# Save
plt.savefig(os.path.join(save_path, "figures", f"heatmap_FS_10_vs_100.png"), bbox_inches='tight')

