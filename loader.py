import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import os
from dataset import *
from setting import *


### Functions for TCGA data
def get_number_features(data):
    for X, y in data:
        n_feat = X.shape[0]
        break
    return n_feat


def get_number_classes(data):
    return len(data.label_key)



### Normalisation   
def transform_data(X, transform='divide_by_sum', factor=1):
    """Return a numpy array or a torch tensor of shape [n, p].
    
    Parameters:
        X  --  Numpy array or torch tensor of shape [n, p] containing n vectors.
        transform  --  Str, name of a transformation in ['log2', 'reverse_log2', 'sqrt', 'reduce_center', 'pearson_regularization', 'divide_by_sum', 'divide_by_norm'].
                       'log2'  --  log2(X + 1)
                       'reverse_log2'  --  2**X - 1
                       'sqrt'  --  sqrt(X)
                       'reduce_center'  --  standardization of each variable i.  (X[:, i] - mean(X[:, i])) / standard deviation(X[:, i]).
                       'pearson_regularization'  --  divide all values of a variable i by the square root of its mean. X[:, i] / sqrt(mean(X[:, i])).
                       'divide_by_sum'  -- each X[k, :] is multiplied by `factor` / sum(X[k, :]). Thus, the sum of the coefficients of X[k, :] is 'factor'.
                       'divide_by_norm'  --  the Euclidean norm of each X[k, :] is set to 1.
        factor  --  Float. Only used when transform in ['divide_by_sum', 'divide_by_norm'].
    """
    assert transform in ['log2', 'reverse_log2', 'sqrt', 'reduce_center', 'pearson_regularization', 'divide_by_sum', 'divide_by_norm'], "transform should be 'log2', 'reverse_log2', 'sqrt', 'reduce_center', 'pearson_regularization', 'divide_by_sum', 'divide_by_norm'"
    # With torch
    if str(X.dtype).split('.')[0] == 'torch':
        if transform == 'log2':
            return torch.log2(X + 1)
        elif transform == 'reverse_log2':
            return 2**X - 1
        elif transform == 'sqrt':
            return torch.sqrt(X)
        elif transform == 'reduce_center':
            mean, std = torch.mean(X, dim=0), torch.std(X, dim=0)
            return (X - mean) / std
        elif transform == 'pearson_regularization':
            mean = torch.mean(X, dim=0)
            return X / torch.sqrt(mean)
        elif transform == 'divide_by_sum':
            return X / torch.sum(X, dim=1).reshape((-1, 1)) * factor 
        elif transform == 'divide_by_norm':
            return X / torch.linalg.norm(X, dim=1).reshape((-1, 1)) * factor
    # With numpy
    else:
        if transform == 'log2':
            return np.log2(X + 1)
        elif transform == 'reverse_log2':
            return 2**X - 1
        elif transform == 'sqrt':
            return np.sqrt(X)
        elif transform == 'reduce_center':
            mean, std = np.mean(X, axis=0), np.std(X, axis=0)
            return (X - mean) / std
        elif transform == 'pearson_regularization':
            mean = np.mean(X, axis=0)
            return X / np.sqrt(mean)
        elif transform == 'divide_by_sum':
            return X / np.sum(X, axis=1).reshape((-1, 1)) * factor
        elif transform == 'divide_by_norm':
            with np.errstate(divide='ignore', invalid='ignore'):  # Trick to enforce division by 0 to be equal to 0.
                norm = np.linalg.norm(X, axis=1).reshape((-1, 1))
                result = X / norm * factor
                result[norm[:, 0] == 0] = 0
            return result
            
 
 
### Normalisation for datasets
def normalize_train_test_sets(X_train, X_test, mean=True, std=True, log2=False, reverse_log2=False, divide_by_sum=False, factor=10**6):
    """Return numpy arrays or a torch tensors X_train, X_test whose values are normalized according to specific transformations.
    
    Parameters:
        X_train  --  Numpy array or torch tensor of shape [n1, p] containing n1 examples.
        X_test  --  Numpy array or torch tensor of shape [n2, p] containing n2 examples.
        mean  --  Remove to each variable p its mean over all training examples.
        std  --  Divide each variable p with each standard deviation computed over all training examples.
        log2  --  log2(X + 1)
        reverse_log2  --  2**X - 1
        divide_by_sum  -- each example X[k, :] is multiplied by `factor` / sum(X[k, :]). Thus, the sum of the coefficients of X[k, :] is 'factor'.
        factor  --  Float. Only used if divide_by_sum is True.
    """
    if reverse_log2:
        X_train = transform_data(X_train, transform='reverse_log2')
        X_test = transform_data(X_test, transform='reverse_log2')
    if divide_by_sum:
        X_train = transform_data(X_train, transform='divide_by_sum', factor=factor)  
        X_test = transform_data(X_test, transform='divide_by_sum', factor=factor)
    if log2:
        X_train = transform_data(X_train, transform='log2')
        X_test = transform_data(X_test, transform='log2')
    if mean:
        mean = X_train.mean(dim=0) if str(X_train.dtype).split('.')[0] == 'torch' else np.mean(X_train, axis=0)
        X_train = X_train - mean
        X_test = X_test - mean
    if std:
        std = X_train.std(dim=0) if str(X_train.dtype).split('.')[0] == 'torch' else np.std(X_train, axis=0)
        X_train = X_train / std
        X_test = X_test / std
    return X_train, X_test



### Normalisation for dataloaders
def find_mean_std(data, train_sampler, device, log2, reverse_log2, divide_by_sum, factor):
    loader = torch.utils.data.DataLoader(data, batch_size=len(train_sampler), sampler=train_sampler)
    X, y = next(iter(loader))
    if reverse_log2:
        X = transform_data(X, transform='reverse_log2')
    if divide_by_sum:
        X = transform_data(X, transform='divide_by_sum', factor=factor)
    if log2:
        X = transform_data(X, transform='log2')
    return X.mean(dim=0).to(device), X.std(dim=0).to(device)


class normalize(nn.Module):
    """Class enabling to normalize the values of the examples.
    """
    def __init__(self, mean, std, log2, reverse_log2, divide_by_sum, factor):
        """
        Parameters:
            mean  --  Torch tensor of shape (1, p). Mean is subtracted to X.
            std  --  Torch tensor of shape (1, p). X is divided by std.
            log2  --  True or False, transform X into log2(X + 1)
            reverse_log2  --  True or False, transform X into 2**X - 1
            divide_by_sum  -- True or False, multiply each example X[k, :] by `factor` / sum(X[k, :]). Thus, the sum of the coefficients of X[k, :] is 'factor'.
            factor  --  Float. Only used if divide_by_sum is True.
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.log2 = log2
        self.reverse_log2 = reverse_log2
        self.divide_by_sum = divide_by_sum
        self.factor = factor
        
    def forward(self, X):
        """
        Parameters:
            X  --  Torch tensor of shape (n, p) containing n examples.
        """
        if self.reverse_log2:
            X = transform_data(X, transform='reverse_log2')
        if self.divide_by_sum:
            X = transform_data(X, transform='divide_by_sum', factor=self.factor)
        if self.log2:
            X = transform_data(X, transform='log2')
        return (X - self.mean) / self.std



### Datasets
# load_data returns two matrices X, y
def load_data(data_path, name, weakly_expressed_genes_removed=True, ood_samples_removed=True, normalize_expression=True):
    """
    Load all examples of the dataset called `name` stored in `data_path` in a data matrix `X` and a label matrix `y`.
    Labels are numbers between 0 and the number of classes. The name of the classes are returned in the `class_name` list.
    The names of the features are returned in the `feat_name` list.
    """
    assert name in ["pancan", "BRCA", "KIRC", "SIMU1", "SIMU2", "SimuA", "SimuB", "SimuC", "demo", "demo1", "ttg-all", "ttg-breast", "BRCA-pam"] or name[:3] == "syn" or name[:3] == "set", "Modify the function load_data to load your own dataset."
    if name in ["pancan", "BRCA", "KIRC", "ttg-all", "ttg-breast", "BRCA-pam"]:
        database, name, label_name = get_TCGA_setting(name)
        data = TCGA_dataset(data_path, database, name, label_name, weakly_expressed_genes_removed, ood_samples_removed, normalize_expression)
        X = np.zeros((len(data), get_number_features(data)))
        y = np.zeros((len(data))).astype('int64')
        for i, (sample, label) in enumerate(data):
            X[i] += sample.numpy()
            y[i] += label.numpy()
        class_name = list(data.label_map.keys())
        feat_name = np.array(data.genes_IDs)
    elif name in ['SIMU1', 'SIMU2', 'SimuA', 'SimuB', 'SimuC', 'demo', 'demo1'] or name[:3] == "syn" or name[:3] == "set":
        data = np.load(os.path.join(data_path, f'{name}.npy'), allow_pickle=True).item()
        X = data['X']
        y = data['y']
        class_name = np.arange(len(np.unique(y)))
        feat_name = np.arange(X.shape[1])
    return X, np.ravel(y), class_name, feat_name
    
    
def split_indices(data_size, test_size, random_state):
    # Shuffle the indices
    indices = list(range(data_size))
    np.random.seed(random_state)
    np.random.shuffle(indices)
    # Split them
    split = int(np.floor(test_size * data_size))
    train_indices, test_indices = indices[split:], indices[:split]
    # Random seed again
    np.random.seed()
    return train_indices, test_indices


def split_data_from_indices(X, y, train_indices, test_indices):
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# load_dataset returns X_train, X_test...
def load_dataset(data_path, name, normalize, regroup=True, classes=None, weakly_expressed_genes_removed=True, ood_samples_removed=True, studied_features=None, normalize_expression=True):
    """
    Load data and split them into a training set and a test set.
    
    Parameters:
        normalize  --  True or False, state whether the data are normalized (each variable is centered and reduced on the training set).
        regroup  --  True or False, used only if the dataset contains subclasses (e.g. 'SimuB', 'SimuC'). If True, subclasses are regrouped into one class. 
                     Otherwise, they appear as different classes.
        classes  --  None or list of integers. If None, all classes are considered. Otherwise, only the elements belonging to the listed classes are kept.
        studied_features  --  List of features or None. If not None, only the listed features are loaded.
    """  
    # Load data
    X, y, class_name, feat_name = load_data(data_path, name, weakly_expressed_genes_removed, ood_samples_removed, normalize_expression)

    # Special case: select particular features
    if studied_features is not None:
        indices = [np.argwhere(feat == feat_name)[0, 0] for feat in studied_features]
        X = X[:, indices]
        feat_name = feat_name[indices]
        assert (feat_name == studied_features).all()

    # Create train/test sets
    test_size, random_state = get_split_dataset_setting(name)
    train_indices, test_indices = split_indices(len(X), test_size, random_state)
    X_train, X_test, y_train, y_test = split_data_from_indices(X, y, train_indices, test_indices)

    # Special case: relabel the classes to create an heterogeneous class
    if regroup:
        if name in ['SimuB', 'SimuC'] or name[:5] == "syn_g":
            for c in range(1, len(np.unique(y)) - 1):
                y_train[np.argwhere(y_train == c)] = np.zeros(np.argwhere(y_train == c).shape)
                y_test[np.argwhere(y_test == c)] = np.zeros(np.argwhere(y_test == c).shape)
            y_train[np.argwhere(y_train == len(np.unique(y)) - 1)] = np.ones(np.argwhere(y_train == len(np.unique(y)) - 1).shape)
            y_test[np.argwhere(y_test == len(np.unique(y)) - 1)] = np.ones(np.argwhere(y_test == len(np.unique(y)) - 1).shape)
            class_name = np.arange(len(np.unique(y_train)))
    
    # Special case: select a subset of classes
    if classes is not None:
        train_indices = [item for _class in classes for item in torch.argwhere(y_train.reshape(-1) == _class)[:, 0].cpu().numpy()]
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        test_indices = [item for _class in classes for item in torch.argwhere(y_test.reshape(-1) == _class)[:, 0].cpu().numpy()]
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]
        
    # Normalize the data        
    if normalize:
        mean, std, log2, reverse_log2, divide_by_sum, factor = get_data_normalization_parameters(name)
        X_train, X_test = normalize_train_test_sets(X_train, X_test, mean, std, log2, reverse_log2, divide_by_sum, factor)
            
    # Information
    n_class = len(np.unique(y_train))
    n_feat = X_train.shape[1]
        
    return X_train, X_test, y_train, y_test, n_class, n_feat, class_name, feat_name   
        

### Loaders
def load_dataloader(data_path, name, device, regroup=True, studied_features=None, weakly_expressed_genes_removed=True, ood_samples_removed=True):
    """
    Function returning torch dataloaders from a dataset called `name`. 

    Parameters:
        regroup  --  True or False. Used only with datasets containing subclasses (e.g. 'SimuA', 'SimuB', 'SimuC'). If regroup, the subclasses are regrouped into one class. Otherwise, they appear as different classes. 
        studied_features  --  List of features or None. If not None, only the listed features are loaded.
    """
    # Setting
    test_size, random_state = get_split_dataset_setting(name)
    use_mean, use_std, log2, reverse_log2, divide_by_sum, factor = get_data_normalization_parameters(name)
    batch_size = get_loader_setting(name)
    
    if name in ["pancan", "ttg-all", "ttg-breast", "BRCA", "KIRC", "BRCA-pam"]:
        # Load data
        database, name, label_name = get_TCGA_setting(name)
        data = TCGA_dataset(data_path, database, name, label_name, weakly_expressed_genes_removed, ood_samples_removed, normalize_expression=True)
        assert len(np.unique(data.genes_IDs)) == len(data.genes_IDs)

        # Selection of features
        if studied_features is not None:
            data.expression = data.expression[studied_features]
            data.genes_IDs = data.expression.columns.values.tolist()

        # Information
        n_class = get_number_classes(data)
        n_feat = get_number_features(data)
        class_name = data.label_key
        feat_name = data.genes_IDs
        n_sample = len(data)
        assert len(feat_name) == n_feat
        
        # Create train/test loaders
        train_indices, test_indices = split_indices(n_sample, test_size, random_state)
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=train_sampler)
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=test_sampler)
        
        # Mean / Std
        mean, std = find_mean_std(data, train_sampler, device, log2, reverse_log2, divide_by_sum, factor)
        
    else:
        # Load data
        X_train, X_test, y_train, y_test, n_class, n_feat, class_name, feat_name = load_dataset(data_path, name, normalize=False, regroup=regroup, weakly_expressed_genes_removed=weakly_expressed_genes_removed, ood_samples_removed=ood_samples_removed, studied_features=studied_features, normalize_expression=True)
        
        # Information
        n_sample = len(X_train) + len(X_test)

        # Create train/test sets
        X_train = torch.from_numpy(X_train).type(torch.float)
        X_test = torch.from_numpy(X_test).type(torch.float)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        train_set = custom_dataset(X_train, y_train)
        test_set = custom_dataset(X_test, y_test)
        
        # Create train/test loaders
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
        
        # Mean / Std
        mean, std = find_mean_std(train_set, np.arange(len(train_set)), device, log2, reverse_log2, divide_by_sum, factor)
        
    # Normalization
    if not use_mean:
        mean = torch.zeros(mean.shape).to(device)
    if not use_std:
        std = torch.ones(std.shape).to(device)
    transform = normalize(mean, std, log2, reverse_log2, divide_by_sum, factor)
        
    return train_loader, test_loader, n_class, n_feat, class_name, feat_name, transform, n_sample



### Useful functions    
def create_balanced_subset_from_data(data, indices, n):
    """
    Return a subset of the list 'indices' containing 'n' examples for each class. 
    data[i] = (X, y), where y is the class (number between 0 and n_class - 1). 
    """
    # Shuffle the indices to return a different list at each execution.
    # Note that the np.random.shuffle shuffles indices outside this function as well. 
    np.random.shuffle(indices)
    # Select the indices associated with the a class until the number of examples 
    # for this class is reached.
    subset_indices = []
    classes = np.zeros(len(data.label_key))
    for i in indices:
        X, y = data[i]
        classes[y] += 1
        if classes[y] <= n:
            subset_indices.append(i)
    return subset_indices


def get_selected_features(selection, selection_type, n_feat_selected, save_path):
    """
    Return an array containing the names of the selected features.

    Parameters:
        selection --  str, name of the selection method used to ranked the features. E.g. var, PCA_PC1, F, MI, L1_exp_1, DESeq2, IG_MLP_set_train_exp_1. 
        selection_type  --  str, best, worst or random_wo_best (random features are selected among all features except the `n_feat_selected` best). 
        n_feat_selected  -- int, number of selected features. If no selection is given, random genes are selected. 
        save_path  --  str, path towards a folder called order containing an array of genes ranked by decreasing order of importance.
    """
    # Check variables
    if selection is not None:
        assert selection_type is not None, "`selection` cannot be used without `selection_type`"
        print(f"Training on the {selection_type} feature for {selection}.") if n_feat_selected == 1 else print(f"Training on {n_feat_selected} {selection_type} features for {selection}.")
    elif n_feat_selected is not None:
        assert selection_type is None, "`selection_type` cannot be used without `selection`"
        print(f"Training on {n_feat_selected} random feature(s).")

    # Select features
    if selection is not None:
        studied_features = np.load(os.path.join(save_path, "order", f"order_{selection}.npy"), allow_pickle=True)
        if selection_type == "best":
            studied_features = studied_features[:n_feat_selected]
        elif selection_type == "worst":
            studied_features = np.flip(studied_features)[:n_feat_selected]
        elif selection_type == "random_wo_best":
            studied_features = studied_features[n_feat_selected:]
            np.random.shuffle(studied_features)
            studied_features = studied_features[:n_feat_selected]
    elif n_feat_selected is not None:
        studied_features = np.array(json.load(open(os.path.join(save_path, "genesIds.txt"))))
        np.random.shuffle(studied_features)
        studied_features = studied_features[:n_feat_selected]
    else:
        studied_features = None
    return studied_features
