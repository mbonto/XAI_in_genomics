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


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Load data
use_DE = True
feat_name = np.array(json.load(open(os.path.join(save_path, "genesIds.txt"))))
n_feat = len(feat_name)
order_var = np.load(os.path.join(save_path, "order", "order_var.npy"))
order_PCA_PC1 = np.load(os.path.join(save_path, "order", "order_PCA_PC1.npy"))
order_F = np.load(os.path.join(save_path, "order", "order_F.npy"))
order_MI = np.load(os.path.join(save_path, "order", "order_MI.npy"))
order_L1 = np.load(os.path.join(save_path, "order", "order_L1_exp_1.npy"))
# order_limma = np.load(os.path.join(save_path, "order", "order_limma.npy"), allow_pickle=True)
if use_DE:
    order_DE = np.load(os.path.join(save_path, "order", "order_DESeq2.npy"), allow_pickle=True) 
order_IG_MLP = np.load(os.path.join(save_path, "order", "order_IG_MLP_set_train_exp_1.npy"), allow_pickle=True) 
order_IG_LR = np.load(os.path.join(save_path, "order", "order_IG_LR_set_train_exp_1.npy"), allow_pickle=True) 
order_IG_GCN = np.load(os.path.join(save_path, "order", "order_IG_GCN_set_train_exp_1.npy"), allow_pickle=True) 


assert len(order_var) == n_feat
assert len(order_PCA_PC1) == n_feat
assert len(order_F) == n_feat
assert len(order_MI) == n_feat
assert len(order_L1) == n_feat
# assert len(order_limma) == n_feat
if use_DE:
    assert len(order_DE) == n_feat
assert len(order_IG_MLP) == n_feat
assert len(order_IG_LR) == n_feat
assert len(order_IG_GCN) == n_feat


###############################################################################################################
############################################ METHOD OVERLAP 1: CURVES #########################################
###############################################################################################################
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


###############################################################################################################
############################################ METHOD OVERLAP 2: HEATMAPS #######################################
###############################################################################################################
n_args = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]

hm = {}
for n in n_args:
    if use_DE:
        data = {'VAR': list(order_var[:n]),
            'PCA': list(order_PCA_PC1[:n]),
            'F': list(order_F[:n]),
            'MI': list(order_MI[:n]),
            # 'chi2': list(order_chi2[:n]),
            #'Limma': list(order_limma[:n]),
            'DE': list(order_DE[:n]),
            'LR+L1': list(order_L1[:n]),
            'MLP': list(order_IG_MLP[:n]),
            'GCN': list(order_IG_GCN[:n]),
            'LR+L2': list(order_IG_LR[:n]),
            # 'Tree': list(order_Tree[:n]),
           }
    else:
        data = {'VAR': list(order_var[:n]),
            'PCA': list(order_PCA_PC1[:n]),
            'F': list(order_F[:n]),
            'MI': list(order_MI[:n]),
            # 'chi2': list(order_chi2[:n]),
            #'Limma': list(order_limma[:n]),
            'LR+L1': list(order_L1[:n]),
            'MLP': list(order_IG_MLP[:n]),
            'GCN': list(order_IG_GCN[:n]),
            'LR+L2': list(order_IG_LR[:n]),
            # 'Tree': list(order_Tree[:n]),
           }
    
    x = [(k1, k2, len(set(d1) & set(d2))) for k1,d1 in data.items() for k2,d2 in data.items()]
    df = pd.DataFrame(x).pivot(index=0, columns=1, values=2)
    if use_DE:
        df = df[['VAR', 'PCA', 'DE', 'F', 'MI', 'MLP', 'GCN', 'LR+L2', 'LR+L1']]  # Limma
        df = df.reindex(['VAR', 'PCA', 'DE', 'F', 'MI', 'MLP', 'GCN', 'LR+L2', 'LR+L1'])  # Limma
    else:
        df = df[['VAR', 'PCA', 'F', 'MI', 'MLP', 'GCN', 'LR+L2', 'LR+L1']]  # Limma
        df = df.reindex(['VAR', 'PCA', 'F', 'MI', 'MLP', 'GCN', 'LR+L2', 'LR+L1'])  # Limma
    if n == 100:
        df_100 = df.copy()
        matrix_triu = np.triu(np.ones(np.shape(df_100)))
        print(n, "features")
        print(df_100.to_numpy())
    elif n == 1000:
        df_1000 = (df.copy() / 10).round().astype(int)
        matrix_tril = np.tril(np.ones(np.shape(df_1000)))
        print(n, "features")
        print(df_1000.to_numpy())
    plt.figure(figsize=(20, 20))
    sns.heatmap(df)
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig(os.path.join(save_path, "figures", f"heatmap_FS_{n}.png"), bbox_inches='tight')


# Heatmap 100 vs 1000 features
plt.figure(figsize=(20, 20))
sns.heatmap(df_100, mask=matrix_triu, vmin=0, vmax=100, cbar=False, annot=True, cmap="BuPu", annot_kws={"fontsize":30})
h = sns.heatmap(df_1000, mask=matrix_tril, vmin=0, vmax=100, annot=True, cmap="BuPu", annot_kws={"fontsize":30}, cbar_kws={"ticks":[0, 100]})
# Labels
plt.xlabel('Upper triangular: comparison among 1000 features', loc='right', labelpad=50, fontsize=30)
plt.ylabel('Lower triangular: comparison among 100 features', loc='bottom', labelpad=50, fontsize=30)
ax = plt.gca()
ax.xaxis.set_label_position('top')
# Ticks
ax.tick_params(axis='both', which='major', labelsize=30)
# Colorbar
cbar = h.figure.axes[-1]
cbar.tick_params(labelsize=30)
cbar.set_yticklabels(['0%', '100%'])
# Save
plt.savefig(os.path.join(save_path, "figures", f"heatmap_FS_100_vs_1000.png"), bbox_inches='tight')
