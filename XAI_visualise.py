import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle
import numpy as np
    

def plot_TCGA_results(results, xlabel, ylabel, save_name=None, show=True, stats=False):
    
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(1, 1, 1)
    
    # Absolute importance per class (decreasing)
    ind = results['indices_cls_best']
    res = results['res_cls_best']['balanced_accuracy']
    ax.plot(ind, res, '-', color='darkorchid', label="Decreasing")
    
    # Balanced absolute importance (decreasing)
    ind = results['indices']
    res = results['res_bal_best']['balanced_accuracy']
    ax.plot(ind, res, '-', color='orchid', label="Decreasing")
    
    # Random
    ind = results['indices']
    res = results['res_rand']['balanced_accuracy']    
    ax.errorbar(ind, res['mean'], fmt='--', yerr=res['std'], color='sienna', label="Random")
    ind = results['indices_cls_best']
    res = results['res_rand_wo_cls_best']['balanced_accuracy']
    ind = ind[:len(res['mean'])]
    ax.plot(ind, res['mean'], ':', color='sienna', label="Random among unimportant")
    
    # Balanced absolute importance (increasing)
    ind = results['indices']
    res = results['res_bal_worst']['balanced_accuracy']
    ax.plot(ind, res, '-', color='orange', label="Increasing")
    
    # Absolute importance per class (increasing)
    ind = results['indices_cls_worst']
    res = results['res_cls_worst']['balanced_accuracy']
    ## Remove the point where all features are set to 0:
    ind = np.array(ind)
    ind[np.argwhere(np.array(ind)==0)] = 1   # enables to keep the continuity of the curve on the plot
    ax.plot(ind, res, '-', color='orangered', label="Increasing")
    
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.legend(fontsize=12)
    plt.xscale('log')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.tight_layout()
    
    # Better order the curves in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    plt.legend([extra, handles[1], handles[3], extra, extra, handles[0], handles[4], extra, handles[5], handles[2]], ["Importance", labels[1], labels[3], " ", "Importance per class", labels[0], labels[4], " ", labels[5], labels[2]])
    
    if save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=150)
        
    if show:
        plt.show()
    
    plt.close('all')
    
        
