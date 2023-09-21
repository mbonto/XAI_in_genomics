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
from feature_selection import *
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
feat_name = np.array(json.load(open(os.path.join(save_path, "genesIds.txt"))))
n_feat = len(feat_name)
order_var = np.load(os.path.join(save_path, "order", "order_var.npy"))
order_PCA_PC1 = np.load(os.path.join(save_path, "order", "order_PCA_PC1.npy"))
order_F = np.load(os.path.join(save_path, "order", "order_F.npy"))
order_MI = np.load(os.path.join(save_path, "order", "order_MI.npy"))
order_L1 = np.load(os.path.join(save_path, "order", "order_L1_exp_1.npy"))
order_limma = np.load(os.path.join(save_path, "order", "order_limma.npy"), allow_pickle=True)
order_DE = np.load(os.path.join(save_path, "order", "order_DESeq2.npy"), allow_pickle=True) 
order_IG = np.load(os.path.join(save_path, "order", "order_IG.npy"), allow_pickle=True) 

assert len(order_var) == n_feat
assert len(order_PCA_PC1) == n_feat
assert len(order_F) == n_feat
assert len(order_MI) == n_feat
assert len(order_L1) == n_feat
assert len(order_limma) == n_feat
assert len(order_DE) == n_feat
assert len(order_IG) == n_feat


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
avg_jaccard_scores_limma = []
avg_jaccard_scores_DE = []
# avg_jaccard_scores_Tree = []

for n in n_args:
    set_IG = set(order_IG[:n])
    set_var = set(order_var[:n])
    set_PCA = set(order_PCA_PC1[:n])
    set_F = set(order_F[:n])
    set_MI = set(order_MI[:n])
    # set_chi2 = set(order_chi2[:n])
    set_L1 = set(order_L1[:n])
    set_limma = set(order_limma[:n])
    set_DE = set(order_DE[:n])
    # set_Tree = set(order_Tree[:n])

    avg_jaccard_scores_var.append(len(list(set_IG.intersection(set_var))) / n)
    avg_jaccard_scores_PCA.append(len(list(set_IG.intersection(set_PCA))) / n)
    avg_jaccard_scores_F.append(len(list(set_IG.intersection(set_F))) / n)
    avg_jaccard_scores_MI.append(len(list(set_IG.intersection(set_MI))) / n)
    # avg_jaccard_scores_chi2.append(len(list(set_IG.intersection(set_chi2))) / n)
    avg_jaccard_scores_L1.append(len(list(set_IG.intersection(set_L1))) / n)
    avg_jaccard_scores_DE.append(len(list(set_IG.intersection(set_DE))) / n)
    avg_jaccard_scores_limma.append(len(list(set_IG.intersection(set_limma))) / n)
    # avg_jaccard_scores_Tree.append(len(list(set_IG.intersection(set_Tree))) / n)


# Save the jaccard scores 
jaccard_all_scores = {'var': avg_jaccard_scores_var, 
                      'PCA': avg_jaccard_scores_PCA, 
                      'F': avg_jaccard_scores_F, 
                      'MI': avg_jaccard_scores_MI, 
                      # 'chi2': avg_jaccard_scores_chi2,
                      'L1': avg_jaccard_scores_L1,
                      'limma': avg_jaccard_scores_limma,
                      'DE': avg_jaccard_scores_DE,}
                      # 'Tree': avg_jaccard_scores_Tree}
create_new_folder(os.path.join(save_path, "figures"))
np.save(os.path.join(save_path, "figures", "jaccard_all_scores.npy"), jaccard_all_scores)


# Plot
plt.figure(figsize=(15, 5))
plt.plot(n_args, avg_jaccard_scores_var, 'x-', label = "vs Var")
plt.plot(n_args, avg_jaccard_scores_PCA, 'x-', label = "vs PCA")
plt.plot(n_args, avg_jaccard_scores_F, 'x-', label = "vs F")
plt.plot(n_args, avg_jaccard_scores_MI, 'x-', label = "vs MI")
#plt.plot(n_args, avg_jaccard_scores_chi2, 'x-', label = "vs chi2")
plt.plot(n_args, avg_jaccard_scores_limma, 'x-', label = "vs Limma")
plt.plot(n_args, avg_jaccard_scores_DE, 'x-', label = "vs DE")
plt.plot(n_args, avg_jaccard_scores_L1, 'x-', label = "vs L1")
#plt.plot(n_args, avg_jaccard_scores_Tree, 'x-', label = "vs Tree")
plt.xscale('log')
plt.xlabel("Number of variables")
plt.ylabel(f"Average Jaccard score \n(IG vs FS methods)")
plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
plt.savefig(os.path.join(save_path, "figures", "jaccard_plots.png"), bbox_inches='tight')


###############################################################################################################
############################################ METHOD OVERLAP 2: HEATMAPS #######################################
###############################################################################################################
n_args = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000]

hm = {}
for n in n_args:
    data = {'Var': list(order_var[:n]),
            'PCA_PC1': list(order_PCA_PC1[:n]),
            'F': list(order_F[:n]),
            'MI': list(order_MI[:n]),
            # 'chi2': list(order_chi2[:n]),
            'Limma': list(order_limma[:n]),
            'DE': list(order_DE[:n]),
            'L1': list(order_L1[:n]),
            'IG': list(order_IG[:n]),
            # 'Tree': list(order_Tree[:n]),
           }
    
    x = [(k1, k2, len(set(d1) & set(d2))) for k1,d1 in data.items() for k2,d2 in data.items()]
    df = pd.DataFrame(x).pivot(index=0, columns=1, values=2)
    print(n, df)
    df = df[['Var', 'PCA_PC1', 'Limma', 'DE', 'F', 'MI', 'L1', 'IG']]
    df = df.reindex(['Var', 'PCA_PC1', 'Limma', 'DE', 'F', 'MI', 'L1', 'IG'])
    print(n, df)
    plt.figure(figsize=(20, 20))
    sns.heatmap(df)
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig(os.path.join(save_path, "figures", f"heatmap_FS_{n}.png"), bbox_inches='tight')
    
