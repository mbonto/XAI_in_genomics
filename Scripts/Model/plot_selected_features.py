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
# from feature_selection import *
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
# order_F = np.load(os.path.join(save_path, "order", "order_F.npy"))
order_MI = {1: np.load(os.path.join(save_path, "order", "order_MI.npy"))}
# order_limma = np.load(os.path.join(save_path, "order", "order_limma.npy"), allow_pickle=True)
order_DE = {1: np.load(os.path.join(save_path, "order", "order_DESeq2.npy"), allow_pickle=True)} 
order_weight_L1 = {}
order_weight_LR = {}
if name in ['temp',]:  # 'ttg-breast', 'BRCA', 'ttg-all']:
    order_effect_L1 = {}
    order_effect_LR = {}
    order_effect_wrt_baseline_L1 = {}
    order_effect_wrt_baseline_LR = {}
order_IG_MLP = {}
order_IG_LR = {}
order_IG_GCN = {}
order_IG_L1 = {}
for exp in range(1, n_repet+1):
    order_weight_L1[exp] = np.load(os.path.join(save_path, "order", f"order_weight_LR_L1_penalty_exp_{exp}.npy"))
    order_weight_LR[exp] = np.load(os.path.join(save_path, "order", f"order_weight_LR_exp_{exp}.npy"))
    if name in ['temp',]:  # 'ttg-breast', 'BRCA', 'ttg-all']:
        order_effect_L1[exp] = np.load(os.path.join(save_path, "order", f"order_effect_LR_L1_penalty_exp_{exp}.npy"))
        order_effect_LR[exp] = np.load(os.path.join(save_path, "order", f"order_effect_LR_exp_{exp}.npy"))
        order_effect_wrt_baseline_L1[exp] = np.load(os.path.join(save_path, "order", f"order_effect_wrt_baseline_LR_L1_penalty_exp_{exp}.npy"))
        order_effect_wrt_baseline_LR[exp] = np.load(os.path.join(save_path, "order", f"order_effect_wrt_baseline_LR_exp_{exp}.npy"))
    order_IG_MLP[exp] = np.load(os.path.join(save_path, "order", f"order_IG_MLP_set_train_exp_{exp}.npy"), allow_pickle=True) 
    order_IG_LR[exp] = np.load(os.path.join(save_path, "order", f"order_IG_LR_set_train_exp_{exp}.npy"), allow_pickle=True) 
    order_IG_GCN[exp] = np.load(os.path.join(save_path, "order", f"order_IG_GCN_set_train_exp_{exp}.npy"), allow_pickle=True) 
    order_IG_L1[exp] = np.load(os.path.join(save_path, "order", f"order_IG_LR_L1_penalty_set_train_exp_{exp}.npy"))


assert len(order_VAR[1]) == n_feat
assert len(order_PCA[1]) == n_feat
# assert len(order_F) == n_feat
assert len(order_MI[1]) == n_feat
# assert len(order_limma) == n_feat
assert len(order_DE[1]) == n_feat
for exp in range(1, n_repet+1):
    assert len(order_weight_L1[exp]) == n_feat
    assert len(order_weight_LR[exp]) == n_feat
    if name in ['temp',]:  # 'ttg-breast', 'BRCA', 'ttg-all']:
        assert len(order_effect_L1[exp]) == n_feat
        assert len(order_effect_wrt_baseline_L1[exp]) == n_feat
        assert len(order_effect_LR[exp]) == n_feat
        assert len(order_effect_wrt_baseline_LR[exp]) == n_feat
    assert len(order_IG_MLP[exp]) == n_feat
    assert len(order_IG_LR[exp]) == n_feat
    assert len(order_IG_GCN[exp]) == n_feat
    assert len(order_IG_L1[exp]) == n_feat


###############################################################################################################
############################################ METHOD OVERLAP 1: CURVES #########################################
###############################################################################################################
"""
n_args = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000]
avg_jaccard_scores_var = []
avg_jaccard_scores_PCA = []
avg_jaccard_scores_F = []
avg_jaccard_scores_MI = []
# avg_jaccard_scores_chi2 = []
avg_jaccard_scores_L1 = []
# avg_jaccard_scores_limma = []
avg_jaccard_scores_DE = []
avg_jaccard_scores_IG_MLP = []
avg_jaccard_scores_IG_GCN = []
# avg_jaccard_scores_Tree = []

for n in n_args:
    set_IG_LR = set(order_IG_LR[:n])
    set_var = set(order_var[:n])
    set_PCA = set(order_PCA_PC1[:n])
    set_F = set(order_F[:n])
    set_MI = set(order_MI[:n])
    # set_chi2 = set(order_chi2[:n])
    set_L1 = set(order_L1[:n])
    # set_limma = set(order_limma[:n])
    if use_DE:
        set_DE = set(order_DE[:n])
    set_IG_MLP = set(order_IG_MLP[:n])
    set_IG_GCN = set(order_IG_GCN[:n])
    # set_Tree = set(order_Tree[:n])

    avg_jaccard_scores_var.append(len(list(set_IG_LR.intersection(set_var))) / n)
    avg_jaccard_scores_PCA.append(len(list(set_IG_LR.intersection(set_PCA))) / n)
    avg_jaccard_scores_F.append(len(list(set_IG_LR.intersection(set_F))) / n)
    avg_jaccard_scores_MI.append(len(list(set_IG_LR.intersection(set_MI))) / n)
    # avg_jaccard_scores_chi2.append(len(list(set_IG_LR.intersection(set_chi2))) / n)
    avg_jaccard_scores_L1.append(len(list(set_IG_LR.intersection(set_L1))) / n)
    if use_DE:
        avg_jaccard_scores_DE.append(len(list(set_IG_LR.intersection(set_DE))) / n)
    avg_jaccard_scores_IG_MLP.append(len(list(set_IG_LR.intersection(set_IG_MLP))) / n)
    avg_jaccard_scores_IG_GCN.append(len(list(set_IG_LR.intersection(set_IG_GCN))) / n)
    # avg_jaccard_scores_limma.append(len(list(set_IG_LR.intersection(set_limma))) / n)
    # avg_jaccard_scores_Tree.append(len(list(set_IG_LR.intersection(set_Tree))) / n)


# Save the jaccard scores 
if use_DE:
    jaccard_all_scores = {'VAR': avg_jaccard_scores_var, 
                      'PCA': avg_jaccard_scores_PCA, 
                      'F': avg_jaccard_scores_F, 
                      'MI': avg_jaccard_scores_MI, 
                      # 'chi2': avg_jaccard_scores_chi2,
                      'LR+L1': avg_jaccard_scores_L1,
                      # 'limma': avg_jaccard_scores_limma,
                      'DE': avg_jaccard_scores_DE,
                      'MLP': avg_jaccard_scores_IG_MLP,
                      'GCN': avg_jaccard_scores_IG_GCN,}
                      # 'Tree': avg_jaccard_scores_Tree}
else:
    jaccard_all_scores = {'VAR': avg_jaccard_scores_var, 
                      'PCA': avg_jaccard_scores_PCA, 
                      'F': avg_jaccard_scores_F, 
                      'MI': avg_jaccard_scores_MI, 
                      # 'chi2': avg_jaccard_scores_chi2,
                      'LR+L1': avg_jaccard_scores_L1,
                      'MLP': avg_jaccard_scores_IG_MLP,
                      'GCN': avg_jaccard_scores_IG_GCN,}
                      # 'limma': avg_jaccard_scores_limma,
                      # 'Tree': avg_jaccard_scores_Tree}
create_new_folder(os.path.join(save_path, "figures"))
np.save(os.path.join(save_path, "figures", "jaccard_all_scores.npy"), jaccard_all_scores)


# Plot
plt.figure(figsize=(15, 5))
plt.plot(n_args, avg_jaccard_scores_var, 'x-', label = "vs VAR")
plt.plot(n_args, avg_jaccard_scores_PCA, 'x-', label = "vs PCA")
plt.plot(n_args, avg_jaccard_scores_F, 'x-', label = "vs F")
plt.plot(n_args, avg_jaccard_scores_MI, 'x-', label = "vs MI")
#plt.plot(n_args, avg_jaccard_scores_chi2, 'x-', label = "vs chi2")
#plt.plot(n_args, avg_jaccard_scores_limma, 'x-', label = "vs Limma")
if use_DE:
    plt.plot(n_args, avg_jaccard_scores_DE, 'x-', label = "vs DE")
plt.plot(n_args, avg_jaccard_scores_L1, 'x-', label = "vs L1")
plt.plot(n_args, avg_jaccard_scores_IG_MLP, 'x-', label = "vs MLP")
plt.plot(n_args, avg_jaccard_scores_IG_GCN, 'x-', label = "vs GCN")
#plt.plot(n_args, avg_jaccard_scores_Tree, 'x-', label = "vs Tree")
plt.xscale('log')
plt.xlabel("Number of variables")
plt.ylabel(f"Average Jaccard score \n(LR+L2 vs other methods)")
plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
plt.savefig(os.path.join(save_path, "figures", "jaccard_plots.png"), bbox_inches='tight')

"""
###############################################################################################################
############################################ METHOD OVERLAP 2: HEATMAPS #######################################
###############################################################################################################
n_args = [10, 100] # 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
hm = {}

for n in n_args:
    data = {
            'VAR': {},
            'PCA': {},
            'MI': {},
            'DE': {},
            'LR+L1 (weight)': {},
            'LR+L2 (weight)':{},
            'LR+L1 (IG)': {},
            'LR+L2 (IG)': {},
            'MLP': {},
            'GNN':{}
            }
    if name in ['temp',]:  # 'BRCA', 'ttg-all', 'ttg-breast']:
        data['LR+L1 (effect)'] = {}
        data['LR+L2 (effect)'] = {}
        data['LR+L1 (effect-b)'] = {}
        data['LR+L2 (effect-b)'] = {}
    data['VAR'][1] = list(order_VAR[1][:n])
    data['PCA'][1] = list(order_PCA[1][:n])
    data['MI'][1] = list(order_MI[1][:n])
    data['DE'][1] =  list(order_DE[1][:n])
    for exp in range(1, n_repet+1):
        data['LR+L1 (weight)'][exp] = list(order_weight_L1[exp][:n])
        data['LR+L2 (weight)'][exp] = list(order_weight_LR[exp][:n])
        data['LR+L1 (IG)'][exp] = list(order_IG_L1[exp][:n])
        data['LR+L2 (IG)'][exp] = list(order_IG_LR[exp][:n])
        data['MLP'][exp] = list(order_IG_MLP[exp][:n])
        data['GNN'][exp] = list(order_IG_GCN[exp][:n])
        if name in ['temp',]: #  'BRCA', 'ttg-all', 'ttg-breast']:
            data['LR+L1 (effect)'][exp] = list(order_effect_L1[exp][:n])
            data['LR+L2 (effect)'][exp] = list(order_effect_LR[exp][:n])
            data['LR+L1 (effect-b)'][exp] = list(order_effect_wrt_baseline_L1[exp][:n])
            data['LR+L2 (effect-b)'][exp] = list(order_effect_wrt_baseline_LR[exp][:n])
       
    x = []
    for k1 in data.keys():
        for k2 in data.keys():
            n_repet_k1 = len(data[k1].keys())
            n_repet_k2 = len(data[k2].keys())
            # print('repet', k1, k2, n_repet_k1, n_repet_k2)
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
            # print('count', k1, k2, count)
            x.append((k1, k2, count))

    # x = [(k1, k2, len(set(d1) & set(d2))) for k1,d1 in data.items() for k2,d2 in data.items()]
    df = pd.DataFrame(x).pivot(index=0, columns=1, values=2)
    if name in ['temp', ]:  # 'BRCA', 'ttg-all', 'ttg-breast']:
        df = df[['VAR', 'PCA', 'MI', 'DE', 'LR+L1 (weight)', 'LR+L1 (effect)', 'LR+L1 (effect-b)', 'LR+L1 (IG)', 'LR+L2 (weight)', 'LR+L2 (effect)', 'LR+L2 (effect-b)', 'LR+L2 (IG)', 'MLP', 'GNN']]
        df = df.reindex(['VAR', 'PCA', 'MI', 'DE', 'LR+L1 (weight)', 'LR+L1 (effect)', 'LR+L1 (effect-b)', 'LR+L1 (IG)', 'LR+L2 (weight)', 'LR+L2 (effect)', 'LR+L2 (effect-b)', 'LR+L2 (IG)', 'MLP', 'GNN'])
    else:
        df = df[['VAR', 'PCA', 'MI', 'DE', 'LR+L1 (weight)', 'LR+L1 (IG)', 'LR+L2 (weight)', 'LR+L2 (IG)', 'MLP', 'GNN']]
        df = df.reindex(['VAR', 'PCA', 'MI', 'DE', 'LR+L1 (weight)', 'LR+L1 (IG)', 'LR+L2 (weight)', 'LR+L2 (IG)', 'MLP', 'GNN'])
    if n == 10:
        df_10 = (df.copy() * 10).round().astype(int)
        matrix_triu = np.triu(np.ones(np.shape(df_10)))
        print(n, "features")
        print(df_10.to_numpy())
    elif n == 100:
        df_100 = (df.copy()).round().astype(int)
        matrix_tril = np.tril(np.ones(np.shape(df_100)), -1)
        print(n, "features")
        print(df_100.to_numpy())
    plt.figure(figsize=(20, 20))
    sns.heatmap(df)
    plt.xlabel('')
    plt.ylabel('')
    # plt.savefig(os.path.join(save_path, "figures", f"heatmap_FS_{n}.png"), bbox_inches='tight')


# Heatmap 10 vs 100 features
plt.figure(figsize=(20, 20))
sns.heatmap(df_10, mask=matrix_triu, vmin=0, vmax=100, cbar=False, annot=True, cmap="BuPu", annot_kws={"fontsize":30}, fmt='d')
h = sns.heatmap(df_100, mask=matrix_tril, vmin=0, vmax=100, annot=True, cmap="BuPu", annot_kws={"fontsize":30}, cbar_kws={"ticks":[0, 100]}, fmt='d')
# Labels
plt.xlabel('Upper triangular: comparison among 100 features', loc='right', labelpad=50, fontsize=30)
plt.ylabel('Lower triangular: comparison among 10 features', loc='bottom', labelpad=50, fontsize=30)
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

