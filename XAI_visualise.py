import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns
    

def plot_TCGA_results(results, xlabel, ylabel, save_name=None, show=True, stats=False):
    
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(1, 1, 1)
    
    # Absolute importance per class (decreasing)
    ind = results['indices_cls_best']
    res = results['res_cls_best']['balanced_accuracy']
    ax.plot(ind, res, '-', color='darkorchid', label="Most important kept first")
    
    # Global absolute importance (decreasing)
    ind = results['indices']
    res = results['res_best']['balanced_accuracy']
    ax.plot(ind, res, '-', color='orchid', label="Most important kept first")
    
    # Random
    ind = results['indices']
    res = results['res_rand']['balanced_accuracy']    
    ax.errorbar(ind, res['mean'], fmt='--', yerr=res['std'], color='sienna', label="Random order")
    ind = results['indices_cls_best']
    res = results['res_rand_wo_cls_best']['balanced_accuracy']
    ind = ind[:len(res['mean'])]
    ax.errorbar(ind, res['mean'], fmt=':', yerr=res['std'], color='sienna', label="Random order without those selected\nas important for the classes")
    # ax.plot(ind, res['mean'], ':', color='sienna', label="Random among unimportant")
    
    # Global absolute importance (increasing)
    ind = results['indices']
    res = results['res_worst']['balanced_accuracy']
    ax.plot(ind, res, '-', color='orange', label="Less important kept first")
    
    # Absolute importance per class (increasing)
    ind = results['indices_cls_worst']
    res = results['res_cls_worst']['balanced_accuracy']
    ## Remove the point where all features are set to 0:
    ind = np.array(ind)
    ind[np.argwhere(np.array(ind)==0)] = 1   # enables to keep the continuity of the curve on the plot
    ax.plot(ind, res, '-', color='orangered', label="Less important kept first")
    
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # ax.legend(fontsize=12)
    plt.xscale('log')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.tight_layout()
    
    # Better order the curves in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    # plt.legend([extra, handles[1], handles[2], extra, extra, handles[0], handles[3], extra, handles[4], handles[5]], ["Global ranking $\phi^g$", labels[1], labels[2], " ", "Ranking per class $\phi^c$", labels[0], labels[3], " ", labels[4], labels[5]])
    
    if save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=150)
        
    if show:
        plt.show()
    
    plt.close('all')


def plot_gene_ranked_by_a_model(set_name_1, set_name_2, n_repet, feat_name, save_path, save_name=None):
    """Plot the distribution of the ranks of the features ordered by two models. For each model, `n_repet` rankings exist. 

    Parameters:
        set_name_1  --  Name of the list of ranked feature names.
        set_name_2  --  Name of the list of ranked feature names.
        n_repet  --  Number of rankings for each model. Stored in set_name_1 + "_exp_1", ..., set_name_1 + "_exp_{n_repet}".
        feat_name  -- List containing feature names.
        save_path  --  Path where the results are stored.
        save_name  --  If not None, the plot is saved in a file called save_name.
    """
    n_feat = len(feat_name)
    plt.figure(figsize=(n_repet, n_repet))
    for exp_1 in range(1, n_repet+1):
        for exp_2 in range(1, n_repet+1):
            if exp_1 < exp_2:
                # List of ranked genes
                gene_list_1 = np.load(os.path.join(save_path, "order", set_name_1 + f"_exp_{exp_1}.npy"), allow_pickle=True)
                gene_list_2 = np.load(os.path.join(save_path, "order", set_name_2 + f"_exp_{exp_2}.npy"), allow_pickle=True)
                # Ranks
                gene_rank_1 = [np.argwhere(feat_name[k] == gene_list_1)[0, 0] for k in range(n_feat)]
                gene_rank_2 = [np.argwhere(feat_name[k] == gene_list_2)[0, 0] for k in range(n_feat)]
                # Plot
                plt.subplot(n_repet - 1, n_repet, exp_2 + (exp_1 - 1) * n_repet)
                sns.kdeplot(x=gene_rank_1, y=gene_rank_2, fill=True)
                if (exp_2 + (exp_1 - 1) * n_repet - 1) % n_repet != exp_1:
                    plt.axis('off')
                if exp_1 == 1:
                    plt.title(exp_2)
                if exp_2 == exp_1 + 1:
                    plt.ylabel(exp_1)
    # Save         
    if save_name is not None:
        plt.savefig(os.path.join(save_path, "figures", save_name), bbox_inches='tight')
    plt.show()


def plot_genes_ranked_by_methods(set_names, methods, feat_name, save_path, save_name):
    """Plot the distribution of the ranks of the features ordered by several methods.

    Parameters:
        set_names  --  List of filenames containing ranked feature names.
        methods  --  List of methods used to rank features.
        feat_name  -- List containing feature names.
        save_path  --  Path where the results are stored.
        save_name  --  If not None, the plot is saved in a file called save_name.
    """
    n_feat = len(feat_name)
    n_set = len(set_names)
    plt.figure(figsize=(n_set*2, n_set*2))
    for i, set_name_1 in enumerate(set_names):
        for j, set_name_2 in enumerate(set_names):
            if i < j:
                # List of ranked genes
                gene_list_1 = np.load(os.path.join(save_path, "order", set_name_1 + ".npy"), allow_pickle=True)
                gene_list_2 = np.load(os.path.join(save_path, "order", set_name_2 + ".npy"), allow_pickle=True)
                # Ranks
                gene_rank_1 = [np.argwhere(feat_name[k] == gene_list_1)[0, 0] for k in range(n_feat)]
                gene_rank_2 = [np.argwhere(feat_name[k] == gene_list_2)[0, 0] for k in range(n_feat)]
                # Plot
                plt.subplot(n_set, n_set, j + 1 + i * n_set)
                sns.kdeplot(x=gene_rank_1, y=gene_rank_2, fill=True)
                if (j + i * n_set) % n_set != (i + 1):
                    plt.axis('off')
                if i == 0:
                    plt.title(methods[j])
                if j == i + 1:
                    plt.ylabel(methods[i])
    if save_name is not None:
        plt.savefig(os.path.join(save_path, "figures", save_name), bbox_inches='tight')
    plt.show()
    
        
