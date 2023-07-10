import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import colorcet as cc
import math
import random
import networkx as nx
from loader import transform_data


def describe_dataset(data):
    # Number of classes
    print(f'The dataset contains {len(data.label_key)} classes. Here are the classes and their respective number of samples:')
    # Number of samples per class
    classes = np.zeros(len(data.label_key))
    for X, y in data:
        classes[y] += 1
    for label in data.label_key:
        index = data.label_map[label]
        print(f'\t{label}: {int(classes[index])}')
    # Number of features
    for X, y in data:
        n_feat = X.shape[0]
        break
    print(f'In total, there are {len(data)} samples, each of them containing {n_feat} features.')


def describe_subset_of_dataset(data, indices):
    # Number of samples per class
    print(f'There are {len(indices)} samples. Here are the classes and their respective number of samples:')
    classes = np.zeros(len(data.label_key))
    for i in indices:
        X, y = data[i]
        classes[y] += 1
    for label in data.label_key:
        index = data.label_map[label]
        print(f'\t{label}: {int(classes[index])}')


def get_min_samples_per_class(data, indices):
    classes = np.zeros(len(data.label_key))
    for i in indices:
        X, y = data[i]
        classes[y] += 1
    return int(min(classes))
        
        
def describe_dataloader(data_loader):
    # Number of samples per class
    classes = np.zeros(len(data_loader.dataset.label_key))
    for X, y in data_loader:
        classes[y] += 1
    for label in data.label_key:
        index = data.label_map[label]
        print(f'\t{label}: {int(classes[index])}')
    # Number of features
    for X, y in data_loader:
        n_feat = X.shape[1]
        break
    print(f'In total, there are {int(np.sum(classes))} samples, each of them containing {n_feat} features.')



def plot_random_gene_expression(data, info_X, info_Y, group_by_classes=True, gene_index=None, log_scale=False, log=False, scale=False, unit='count'):
    if scale:
        info_X = np.log2(transform_data(2**info_X-1, transform='divide_by_sum', factor=10**6) + 1)
    
    if log:
        info_X = np.log2(info_X + 1)
    
    if gene_index is None:
        gene_index = np.random.randint(info_X.shape[1])

    plt.figure(figsize=(20, 3))
    
    if group_by_classes:
        sns.boxplot(x=[data.inv_label_map[index] for index in info_y], y=info_X[:, gene_index], order=np.sort(np.unique([data.inv_label_map[index] for index in info_y])))
        plt.title(f"Expression of gene {data.genes_IDs[gene_index]} ({gene_index}) grouped by class in the dataset.")
        plt.tick_params(axis='both', which='major', labelsize=11)
        if log_scale:
            plt.yscale('log')
        plt.ylabel(f"Gene expresion\n({unit})")
    else:
        sns.boxplot(x=info_X[:, gene_index])
        plt.title(f"Expression of gene {data.genes_IDs[gene_index]} ({gene_index}) in the dataset.")
        if log_scale:
            plt.xscale('log')
        plt.xlabel(f"Gene expresion ({unit})")


def plot_random_sample_expression(data, log=False, unit='count', index=None, scale=False):
    plt.figure(figsize=(20, 2))
    if index is None:
        index = np.random.randint(0, len(data))
    ID = data.sample_IDs[index]
    if scale:
        sample = transform_data((2**data.expression.loc[ID].values-1).reshape(1, -1), transform='divide_by_sum', factor=10**6).reshape(-1)
        plt.plot(np.log2(sample + 1), '.')
    elif log:
        plt.plot(np.log2(data.expression.loc[ID].values + 1), '.')
    else:
        plt.plot(data.expression.loc[ID].values, '.')
    plt.xlabel("Gene index")
    plt.ylabel(f"Gene expression\n({unit})")
    plt.title("Gene expression in a random sample.")         
        

def plot_stats_on_gene_expression(data, info_X, info_Y, criteria='average', log_scale=False, log=False, scale=False, unit='count'):    
    if scale:
        info_X = np.log2(transform_data(2**info_X-1, transform='divide_by_sum', factor=10**6) + 1)
    
    if log:
        info_X = np.log2(info_X + 1)
        
    if criteria == 'average':
        info_X = np.mean(info_X, axis=0)
        print(f"There are {np.sum(info_X == 0)} genes whose average expression is 0.")
    elif criteria == 'median':
        info_X = np.median(info_X, axis=0)
        print(f"There are {np.sum(info_X == 0)} genes whose median expression is 0.")
    elif criteria == 'std':
        info_X = np.std(info_X, axis=0)
        print(f"There are {np.sum(info_X == 0)} genes whose standard deviation is 0.")
    elif criteria == 'min':
        info_X = np.min(info_X, axis=0)
        print(f"There are {np.sum(info_X == 0)} genes whose minimum is 0.")
    elif criteria == 'max':
        info_X = np.max(info_X, axis=0)
        print(f"There are {np.sum(info_X == 0)} genes whose maximum is 0.")
    
    plt.figure(figsize=(20, 2))
    plt.plot(info_X, '.')
    plt.xlabel("Gene index")
    plt.ylabel(f"{criteria.capitalize()} expression\n({unit})")
    plt.show()

    plt.figure(figsize=(20, 2))
    sns.boxplot(x=info_X)
    if log_scale:
        plt.xscale('log')
    plt.xlabel(f"{criteria.capitalize()} expression per locus ({unit})")
    
    
def sort_genes(data, info_X, info_Y, criteria='average', log_scale=False, scale=False, unit='count'):
    if scale:
        info_X = np.log2(transform_data(2**info_X-1, transform='divide_by_sum', factor=10**6) + 1)
    
    if criteria == 'average':
        info_X = np.mean(info_X, axis=0)
    elif criteria == 'median':
        info_X = np.median(info_X, axis=0)
    elif criteria == 'std':
        info_X = np.std(info_X, axis=0)
    elif criteria == 'min':
        info_X = np.min(info_X, axis=0)
    elif criteria == 'max':
        info_X = np.max(info_X, axis=0)
    
    plt.figure(figsize=(20, 5))
    plt.plot(np.sort(info_X), '.')
    plt.xlabel(f"Genes sorted by {criteria.capitalize()} expression")
    plt.ylabel(f"{criteria.capitalize()} expression ({unit})")
    if log_scale:
        plt.yscale("log")
    plt.show()
    
    return np.argsort(info_X)
    

def plot_class_imbalance(data, label_name, save_path=None):
    classes = np.zeros(len(data.label_key))
    for X, y in data:
        classes[y] += 1
    xlabels = {"type": "Cancer class", "sample_type.samples": "Type"}
    plt.figure(figsize=(20, 3))
    sns.barplot(x=[label for label in data.label_key], y=[classes[data.label_map[label]] for label in data.label_key], order=np.sort([label for label in data.label_key]))
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.xlabel(f"{xlabels[label_name]}", fontsize=18)
    plt.ylabel("Number of samples", fontsize=18)
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "class_imbalance"), bbox_inches='tight')
    plt.show()
    

def describe_gene_expression(data, info_X, info_Y, log_scale=True, unit='count', log=False, scale=False):
    if scale:
        info_X = np.log2(transform_data(2**info_X-1, transform='divide_by_sum', factor=10**6) + 1)
        
    if log:
        info_X = np.log2(info_X + 1)
    
    print("Mean: ", np.round(np.mean(info_X), 2))
    print("Median: ", np.round(np.median(info_X), 2))
    print("Max: ", np.round(np.max(info_X), 2))
    print("Min: ", np.round(np.min(info_X), 2))

    plt.figure(figsize=(20, 2))
    if log:
        sns.histplot(info_X.reshape(-1) + np.min(info_X), log_scale=False)
    else:
        sns.histplot(info_X.reshape(-1) + 1, binrange=[0, 7], bins=24, log_scale=True)
    plt.xlabel(f"Distribution of gene expression ({unit}) across the dataset")
    plt.ylabel("Number of\ngenes")
    if log_scale:
        plt.yscale('log')
    plt.show()
    
    plt.figure(figsize=(20, 2))
    _sum = np.sum(info_X, axis=1)
    print('Below, the gene expressions are summed per individual.')
    if log:
        sns.histplot(_sum, log_scale=False)
    else:
        sns.histplot(_sum, log_scale=False)
    plt.xlabel(f"Sum of gene expressions for each individual sample ({unit})")
    plt.ylabel("Number of\nindividuals")
    if log_scale:
        plt.yscale('log')
    plt.show()
    

def do_scatterplot_2D(X, y, labels, xlabel=None, ylabel=None, dim1=0, dim2=1, legend=True, size=2, ticks=True, save_name=None):
    classes = np.unique(y)
    cmap = sns.color_palette(palette=cc.glasbey, n_colors=len(classes))
    
    plt.figure(figsize=(4, 4))
    for i, _class in enumerate(classes):
        plt.scatter(X[y[:, 0]==_class, dim1], X[y[:, 0]==_class, dim2], color=cmap[i], label=labels[_class], s=size)
    if legend:
        plt.legend(ncol=1, bbox_to_anchor=(1, 0.9), markerscale=4.)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if not ticks:
        plt.xticks(labels=[], ticks=[])
        plt.yticks(labels=[], ticks=[])
    if save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=150)
    plt.show()
    
    

def plot_variables_in_boxes(X, feat_name, n_feat=10, method='max', save_name=None):
    n_sample = X.shape[0]
    plt.figure(figsize=(20, 3))
    if method == 'max':
        V = np.argsort(-np.max(X, axis=0))[:n_feat]
    elif method == 'min':
        V = np.argsort(np.max(X, axis=0))[:n_feat]
    elif method == 'mean_abs_max':
        V = np.argsort(-np.mean(np.abs(X), axis=0))[:n_feat]
    elif method == 'mean_max':
        V = np.argsort(-np.mean(X, axis=0))[:n_feat]
    elif method == 'mean_min':
        V = np.argsort(np.mean(X, axis=0))[:n_feat]
    elif method == 'median_max':
        V = np.argsort(-np.median(X, axis=0))[:n_feat]
        print(-np.sort(-np.median(X, axis=0))[:n_feat])
    elif method == 'median_min':
        V = np.argsort(np.median(X, axis=0))[:n_feat]
    sns.boxplot(x=[name for name in feat_name[V] for sample in range(n_sample)], y=[X[sample, v] for v in V for sample in range(n_sample)], order=[name for name in feat_name[V]])
    plt.title(f"Values of the variables across samples (a point = a sample)")
    plt.tick_params(axis='both', which='major', labelsize=11)
    if save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=150)
    plt.show()
    
    
def plot_box(data, xlabel=None, save_name=None):
    plt.figure(figsize=(15, 4))
    plt.boxplot(data, vert=False)
    plt.yticks(color='w')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    if xlabel:
        plt.xlabel(xlabel)
    if save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=150)
    plt.show()
    
    
def describe_random_individuals(data, info_X, info_Y, log=True, scale=False, log_scale=False, save_path=None, unit='log2(count+1)'):
    if scale:
        info_X = np.log2(transform_data(2**info_X-1, transform='divide_by_sum', factor=10**6) + 1)
    if log:
        info_X = np.log2(info_X+1)

    plt.figure(figsize=(15, 4))

    my_samples = {}
    for i in range(20):
        j = np.random.randint(info_X.shape[0])
        my_samples[i] = info_X[j]

    plt.boxplot(my_samples.values(), vert=True)
    if log_scale:
        plt.yscale('log')
    plt.xticks(color='w')
    plt.xlabel('Distribution of gene expression in 20 individuals (boxes)')
    plt.ylabel(f'Gene expression\n({unit})')
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "random_individuals"), bbox_inches='tight')
    plt.show()
    

def draw_from_Dirichlet(param):
    '''
    Draw a realisation of a Dirichlet distribution parametrized by `param`.
    
    Parameters:
        param  --  np vector, used to parametrized the Dirichlet distribution.
    '''
    print(f"    Parameter of the Dirichlet distribution: {param}")
    r = param.copy()
    values = np.random.dirichlet(r[r!=0])
    r[r!=0] = values
    print(f"    Realisation: {values / np.sum(values)}")
    return r



def draws_from_Dirichlet(dict_param, label_key):
    '''
    Draw the realisations of sevral Dirichlet distributions parametrized by the parameters in `dict_param`.
    
    Parameters:
        dict_param  --  dict. dict[key] contains a vector used to parametrized a Dirichlet distribution.
        label_key  --  str, name of the keys represented by each parameter vector.
    '''
    r = {}
    # For each param in dict_param, draw a realisation r
    for c, key in enumerate(dict_param.keys()):
        print(f'{label_key} {key}')
        r[key] = draw_from_Dirichlet(dict_param[key])
    return r


def plot_draws_from_Dirichlet(dict_real, label_key, label_var, save_name):
    '''
    Plot the realisations of Dirichlet distributions in a barplot.
    
    Parameters:
        dict_real  --  dict. dict_real[key] contains a draw from a Dirichlet distribution.
        label_key  --  str, name of the keys represented by each parameter vector.
        label_var  --  str, name of the variables represented by the values of the parameter vector.
        save_name  -- str or None, path where the plot will be stored.
    '''
    # Parameters
    n_var = 0
    for key in dict_real.keys():
        n_var = max(n_var, len(dict_real[key]))
    
    # Colors attributed to each variable
    cmap = cm.get_cmap('viridis', n_var)
    colors = np.arange(n_var)
    np.random.seed(0)
    np.random.shuffle(colors)
    np.random.seed()
    colors = [cmap(c) for c in list(colors)]
    
    # Figure
    fig, ax = plt.subplots()
    # For each param in dict_param...
    for c, key in enumerate(dict_real.keys()):
        # Get a realisation r...
        r = dict_real[key]
        # Create a barplot, band by band
        label = f'{label_var} {0}' if c==0 else None
        ax.barh(f'{label_key} {c}', r[0], color=colors[0], label=label)
        bottom = r[0]
        for var in range(1, n_var):
            label = f'{label_var} {var}' if c==0 else None
            ax.barh(f'{label_key} {c}', r[var], left=bottom, color=colors[var], label=label)
            bottom += r[var]
    # Legend
    ax.set_ylabel('Proportions')
    ax.set_xticklabels([])
    ax.invert_yaxis()
    if n_var < 5:
        ax.legend()
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()


def plot_average_signals(X, y, n_variable=None, save_name=None):
    n_class = len(np.unique(y))
    colors = sns.color_palette(None, n_class)
    plt.figure(figsize=(10, 1))
    for c in range(n_class):
        mean = np.mean(X[y==c], axis=0)
        std = np.std(X[y==c], axis=0)
        plt.errorbar(np.arange(len(mean))[:n_variable], mean[:n_variable], yerr=std[:n_variable], color=colors[c], label=f"class {c}")
    plt.xlabel("Variables")
    plt.ylabel("Values")
    plt.legend(ncol=math.ceil(n_class/2), loc='upper center', bbox_to_anchor=(0.5, 1.7), fancybox=True, shadow=False)
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()
    
    
def plot_random_signals(X, y, n_sample_per_class=1, n_variable=None, save_name=None, legend=True):
    n_class = len(np.unique(y))
    colors = sns.color_palette(None, n_class)
    count_per_class = np.zeros(n_class)
    plt.figure(figsize=(10, 1))
    indices = np.arange(X.shape[0])
    random.shuffle(indices)
    i = 0
    while np.sum(count_per_class) != n_sample_per_class * n_class and i != X.shape[0]:
        if count_per_class[y[indices[i]]] < n_sample_per_class:
            if count_per_class[y[indices[i]]] == 0:
                label = f"class {y[indices[i]]}"
            else:
                label = None
            plt.plot(X[indices[i], :n_variable], color=colors[y[indices[i]]], label=label)
            count_per_class[y[indices[i]]] += 1
        i += 1
    plt.xlabel("Variables")
    plt.ylabel("Values")
    if legend:
        plt.legend(ncol=math.ceil(n_class/2), loc='upper center', bbox_to_anchor=(0.5, 1.7), fancybox=True, shadow=False)
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()
    

    
def print_average_signals(X, y, var_list=[0, 1, 2]):
    n_class = len(np.unique(y))
    print(f"Average signals +- standard deviations for variables in {var_list}")
    for c in range(n_class):
        mean = np.mean(X[y==c], axis=0)
        std = np.std(X[y==c], axis=0)
        print(f"    Class {c}  - {np.round(mean[var_list])}  +-  {np.round(std[var_list])}")
        

def plot_matrix(A, xlabel, ylabel, save_name=None):
    plt.matshow(A)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()


def draw_a_graph_representing_data_simulated_with_LDA(n_class, n_pathway, n_gene, alpha, eta, node_size=10, font_size=1, save_name=None):
    # Nx graph
    G = nx.Graph()
    pos = {}

    # Nodes
    center = ((n_class+1)%2)/2
    for c in range(n_class):
        G.add_node('class_' + str(c))
        pos['class_' + str(c)] = [(- int(n_class / 2) + c + center)*5, 2]

    center = - ((n_pathway+1)%2)/2
    for p in range(n_pathway):
        G.add_node('path_' + str(p))
        pos['path_' + str(p)] = [(- int(n_pathway / 2) + p + center)*5, 1]

    center = - (n_gene%2)/2
    for g in range(n_gene):
        G.add_node('G_' + str(g))

    # Edges
    for c in range(n_class):
        for p in range(n_pathway):
            G.add_edge('class_' + str(c), 'path_' + str(p), weight=alpha['C'+str(c)][p]/100) 
    for p in range(n_pathway):
        for g in range(n_gene):
            G.add_edge('path_' + str(p), 'G_' + str(g), weight=eta['P'+str(p)][g]/np.sum(eta['P'+str(p)])) 

    # Position
    weight = {}
    for p in range(n_pathway):
        weight[str(p)] = []
        pos[str(p)] = []
        for g in range(n_gene):
            weight[str(p)].append(G['G_' + str(g)]['path_' + str(p)]['weight'])
        weight[str(p)] = np.array(weight[str(p)])

    set1 = set(list(np.argwhere(weight[str(0)] != 0).reshape(-1)))
    set2 = set(list(np.argwhere(weight[str(1)] != 0).reshape(-1)))
    set3 = set(list(np.argwhere(weight[str(2)] != 0).reshape(-1)))
    inter12 = set1.intersection(set2)
    inter13 = set1.intersection(set3).difference(inter12)
    inter23 = set2.intersection(set3).difference(inter13)
    tot = set(np.arange(n_gene))
    tot = tot.difference(set1)
    tot = tot.difference(set2)
    tot = tot.difference(set3)
    order = list(set1 - inter12 - inter13) + list(inter12) + list(inter13) + list(set2 - inter12 - inter13 - inter23) + list(inter23) + list(set3 - inter12 - inter13 - inter23) + list(tot)

    center = - ((n_gene+1)%2)/2
    for i, g in enumerate(order):
        pos['G_' + str(g)] = [- int(n_gene / 2) + i + center, 0]
        
    # Labels
    labels = {}
    for item in list(G.nodes):
        labels[item] = item
    
    # Draw
    plt.figure(figsize=(13, 3))
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightgreen')
    widths = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edges(G, pos, edgelist=widths.keys(), width=list(widths.values()))
    nx.draw_networkx_labels(G, pos, labels, font_size=font_size, font_color='purple')
    
    if save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=150)
        
    plt.show()
    
