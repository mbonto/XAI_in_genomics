import numpy as np
import torch
import os
from sklearn.metrics import classification_report, balanced_accuracy_score
import warnings
from evaluate import *
from loader import *


# Useful functions
def sort_attribution_scores(attr):
    return np.argsort(-np.abs(attr))

    
def get_attributions_per_class(attr, labels, _class, method=None, value=None):
    """
    Return the normalized attributions associated with the examples associated with the label _class.
    """
    indices = np.argwhere(labels == _class)[:, 0]
    attr_cls = attr[indices]
    return normalize_attribution_scores(attr_cls, method, value)


def normalize_attribution_scores(attr, method, value=None):
    """
    To globally interpret local explanations, the scores are normalized per feature over all inputs.
    """
    assert method in ['mean', 'quantile'], "`method` must be 'mean' or 'quantile'"
    if method == 'quantile':
        assert value is not None and value >=0 and value <= 1, "when method=='quantile', the value must be a floating value between 0 and 1."
        attr = torch.quantile(attr, value, dim=0)
    elif method == 'mean':
        attr = attr.mean(0)
    return attr


def get_number_common_elements(list1, list2):
    return len(list(set(list1).intersection(list2)))


def get_common_genes(list1, list2, interval):
    nb = []
    for gap in interval:
        nb.append(get_number_common_elements(list1[:gap], list2[:gap]))
    return nb


def sort_features(reference, order, scores):
    """
    Return the indices of the features ordered by importance for a given reference.
    
    Parameters:
        reference  --  "general" or a number between 0 and 32 representing a class
        order  -- "increasing", features ordered with increasing importance (default: decreasing)
    """
    sorted_indices = scores[reference]['sorted_indices']  
    if order == 'increasing':
        sorted_indices = sorted_indices[::-1].copy()
    return sorted_indices


def remove_features(X, sorted_features, number, baseline):
    """Replace the first `number` sorted_features in `X` by th values in `baseline`.
    """
    zeros = sorted_features[:number]
    X_temp = X.clone()
    device = X_temp.device
    s = np.repeat(np.arange(X.shape[0]), len(zeros))
    f = np.tile(zeros, X.shape[0])
    X_temp[s, f] = baseline[np.repeat(np.zeros(X.shape[0]), len(zeros)), f].clone().detach().to(device)
    return X_temp


def keep_features(X, best_indices, baseline):
    """Set to 0 the features which are not in best_indices.
    
    Parameters:
        X  --  Inputs, torch tensor (n_sample, n_feat).
        best_indices  -- Indices of the features to keep (the others are set to 0).
        baseline  --  Values used to modify the values of the input features. Tensor (1, n_feat).
    """
    zeros = np.array(list(set(np.arange(0, X.shape[1], 1)) - set(best_indices)))
    X_temp = X.clone()
    device = X_temp.device
    s = np.repeat(np.arange(X.shape[0]), len(zeros))
    f = np.tile(zeros, X.shape[0])
    X_temp[s, f] = baseline[np.repeat(np.zeros(X.shape[0], dtype='int64'), len(zeros)), f].clone().detach().to(device)
    return X_temp
            
            
def get_metrics(model, X, y, labels_name=None, get_cm=True):
    """Given a model, input data X and labels y, return the accuracy, the classification report and a confusion matrix.    
    """
    outputs = model(X)
    _, y_pred = torch.max(outputs.data, 1)
    y_pred = y_pred.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    acc = compute_accuracy_from_predictions(y_pred, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b_acc = np.round(balanced_accuracy_score(y, y_pred)*100, 2)
    _dict = classification_report(y, y_pred, output_dict=True, zero_division=0)
    if get_cm:
        cm = get_confusion_matrix(y, y_pred, labels_name, normalize='true')
        return b_acc, acc, _dict, cm
    else:
        return b_acc, acc, _dict


def get_metrics_with_removed_features(model, X, y, baseline, classes, sorted_features, nb_to_remove, feat_to_remove=[]):
    """
    After removing some features, the accuracy, precision, recall are computed. 
    
    Parameters:
        model  --  neural network
        X  --  input, torch tensor (batch_size, n_feat)
        y  --  labels, torch tensor (batch_size)
        baseline  --  Values used to modify the values of the input features. Tensor (1, n_feat).
        classes  -- list containing the classes to consider
        sorted_features  -- indices of the features, the first ones are removed first.
        nb_to_remove -- list containing the numbers of features to remove from X
        feat_to_remove  --  [], list containing lists of features to remove from X (the additional ones are removed randomly)
    """
    balanced_accuracy = []
    accuracy = []
    recall = {}
    precision = {}
    
    for _class in classes:
        recall[_class] = []
        precision[_class] = []
        
    for i, nb in enumerate(nb_to_remove):
        if feat_to_remove != []:
            features = list(feat_to_remove[i]) + [feat for feat in sorted_features if feat not in feat_to_remove[i]]
        else:
            features = sorted_features.copy()
        X_temp = remove_features(X, features, nb, baseline)
        b_acc, acc, _dict = get_metrics(model, X_temp, y, None, get_cm=False)
        balanced_accuracy.append(b_acc)
        accuracy.append(acc)
        
        for _class in classes:
            recall[_class].append(_dict[str(_class)]['recall'])
            precision[_class].append(_dict[str(_class)]['precision'])
        
    return balanced_accuracy, accuracy, recall, precision



def get_metrics_with_selected_features(model, X, y, feat_to_keep, baseline, classes=[]):
    """
    Return the performance of the model after setting some features to 0. 
    
    Parameters:
        model  --  Neural network.
        X  --  Inputs, torch tensor (n_sample, n_feat).
        y  --  Labels, torch tensor (n_sample).
        feat_to_keep  -- Indices of the features to keep (the others are set to 0).
        baseline  --  Values used to modify the values of the input features. Tensor (1, n_feat).
        classes  --  Classes to consider (for precision and recall). List or None.
    """
    recall = {}
    precision = {}

    X_temp = keep_features(X, feat_to_keep, baseline)
    balanced_accuracy, accuracy, _dict = get_metrics(model, X_temp, y, None, get_cm=False)

    for _class in classes:
        recall[_class] = _dict[str(_class)]['recall']
        precision[_class] = _dict[str(_class)]['precision']
        
    return balanced_accuracy, accuracy, recall, precision



# Global representation (per class)
def get_results_per_class(model, X, y, K, scores, setting, baseline, classes=[]):
    """
    Compute several metrics measuring the performance of 'model' when the top-k features per class are kept (others features set to 0) or removed (set to 0).
    Return 
        n_kept  --  a list with the number of features kept
        kept_feat  --  a list with the indices of features kept
        results  --  a dictionary containing
            results['balanced_accuracy']  --  a list of balanced accuracy
            results['accuracy'] --  a list of accuracy
            results[_class]['recall']  --  a list of recall for a _class specified in `classes`
            results[_class]['precision']  --  a list of precision for a _class specified in `classes`
    
    Parameters:
        model  --  neural network
        X  --  input, torch tensor (batch_size, n_feat)
        y  --  labels, torch tensor (batch_size)
        K -- list of integers k, numbers of best features per class to keep or remove
        scores  --  dict, scores[_class] contains two lists scores[_class]['attr'], scores[_class]['sorted_indices'].
                    scores[_class]['attr'] is a list of floats indicating the importance of a feature.
                    scores[_class]['sorted_indices'] is a list of indices sorting the 'attr' with decreasing absolute values.
        setting  --  'keep_best' or 'remove_best', indicate whether the best features are kept or removed
        baseline  --  Values used to modify the values of the input features. Tensor (1, n_feat).
        classes  -- None or list containing the classes to consider (for precision and recall).
    """
    assert setting in ['keep_best', 'remove_best']
    
    n_kept = []
    kept_feat = []
    results = {}
    results['accuracy'] = []
    results['balanced_accuracy'] = []
    for _class in classes:
        results[_class] = {}
        results[_class]['recall'] = []
        results[_class]['precision'] = []
        
    for k in K:
        
        if setting == 'keep_best':
            feat_to_keep = keep_best_features(k, scores)
        else:
            feat_to_keep = list(set(np.arange(X.shape[1])) - set(keep_best_features(k, scores)))
             
        n_kept.append(len(feat_to_keep))
        kept_feat.append(feat_to_keep)

        balanced_accuracy, accuracy, recall, precision = get_metrics_with_selected_features(model, X, y, feat_to_keep, baseline, classes)

        results['accuracy'].append(accuracy)
        results['balanced_accuracy'].append(balanced_accuracy)

        for _class in classes:
            results[_class]['recall'].append(recall[_class])
            results[_class]['precision'].append(precision[_class])

    return results, n_kept, kept_feat


def keep_best_features(nb_per_class, scores):
    """
    Return the indices of the nb_per_class most important features.
    
    Parameters:
        nb_per_class -- number of features to keep per class
        scores  --  dict, scores[_class] contains two lists scores[_class]['attr'], scores[_class]['sorted_indices'].
                    scores[_class]['attr'] is a list of floats indicating the importance of a feature.
                    scores[_class]['sorted_indices'] is a list of indices sorting the 'attr' with decreasing absolute values.
    """
    n_class = len(scores.keys()) - 1
    indices = []
    for _class in range(n_class):
        indices.extend(np.argsort(-scores[_class]['attr'])[:nb_per_class])  ## scores[_class]['sorted_indices'][:nb_per_class])
    return np.unique(indices)



# Global representation (all studied examples)
def get_results_with_best_features_kept_or_removed(model, X, y, K, attr, baseline, classes=[], kept=True, balance=False):
    """
    Compute several metrics measuring the performance of 'model' when the top-k best features are removed (set to 0) or 
    kept (all the others set to 0).
    
    Return 
        kept_feat  --  a list with the indices of features kept
        results  --  a dictionary containing
            results['balanced_accuracy']  --  a list of balanced accuracy
            results['accuracy'] --  a list of accuracy
            results[_class]['recall']  --  a list of recall for a _class specified in `classes`
            results[_class]['precision']  --  a list of precision for a _class specified in `classes`
            
    Parameters:
        model  --  Neural network.
        X  --  Input, torch tensor (batch_size, n_feat).
        y  --  Labels, torch tensor (batch_size).
        K --  Numbers of best features to keep or remove. List of integers k.
        attr  --  Importance of the features per sample, array (batch_size, n_feat).
        baseline  --  Values used to modify the values of the input features. Tensor (1, n_feat).
        classes  --  Classes to consider (for precision and recall). None or list.
        kept  --  Indicate whether the features are kept or removed. True or False,
        balance  -- Indicate whether the average importance of a feature is balanced wrt the class imbalance. True or False.
    """
    kept_feat = []
    results = {}
    results['accuracy'] = []
    results['balanced_accuracy'] = []
    for _class in classes:
        results[_class] = {}
        results[_class]['recall'] = []
        results[_class]['precision'] = []
    
    if balance:
        n_class = len(torch.unique(y))
        n_feat = X.shape[1]
        scores = np.zeros(n_feat)
        for _class in range(n_class):
            scores += np.mean(attr[y.cpu() == _class], axis=0)
            ## scores += np.mean(np.abs(attr[y.cpu() == _class]), axis=0)
        scores = scores / n_class
    else:
        scores = np.mean(attr, axis=0)
        # scores = np.mean(np.abs(attr), axis=0)
    
    if kept:
        indices = np.argsort(-scores)
    else:
        indices = np.argsort(scores)
    
    for k in K:
        feat_to_keep = indices[:k]
        kept_feat.append(feat_to_keep)
        balanced_accuracy, accuracy, recall, precision = get_metrics_with_selected_features(model, X, y, feat_to_keep, baseline, classes)

        results['accuracy'].append(accuracy)
        results['balanced_accuracy'].append(balanced_accuracy)
            
        for _class in classes:
            results[_class]['recall'].append(recall[_class])
            results[_class]['precision'].append(precision[_class])

    return results, kept_feat



# Random
def get_results_with_random_features(model, X, y, nb_to_keep, n_simu, baseline, classes=[], feat_to_remove=[]):
    """
    Return the accuracy and balanced accuracy of 'model' when random features are set to 0. 
    Also return the precision and recall for the specified 'classes'.
    
    Parameters:
        model  --  Neural network.
        X  --  Input, torch tensor (batch_size, n_feat).
        y  --  Labels, torch tensor (batch_size).
        nb_to_keep -- Numbers of features to keep from X. List.
        baseline  --  Values used to modify the values of the input features. Tensor (1, n_feat).
        n_simu  --  Number of simulations the results are computed on. Integer.
        classes  -- Classes to consider (for precision and recall). List or None.
        feat_to_remove  -- Lists of features to remove from X (the additional ones are removed randomly). List or None.
    """
    # Initialize the outputs
    results = {}
    results['balanced_accuracy'] = []
    results['accuracy'] = []
    for _class in classes:
        results[_class] = {}
        results[_class]['recall'] = []
        results[_class]['precision'] = []
    
    # Compute the outputs for n_simu
    n_feat = X.shape[1]
    random_features = np.arange(0, n_feat, 1)
    for i in range(n_simu):
        np.random.shuffle(random_features)
        balanced_accuracy, accuracy, recall, precision = get_metrics_with_removed_features(model, X, y, baseline, classes, random_features, n_feat - nb_to_keep, feat_to_remove)
        results['balanced_accuracy'].append(balanced_accuracy)
        results['accuracy'].append(accuracy)
        for _class in classes:
            results[_class]['recall'].append(recall[_class])
            results[_class]['precision'].append(precision[_class])
    
    # Average the outputs
    final_results = {}
    for term in ['balanced_accuracy', 'accuracy']:
        final_results[term] = {}
        final_results[term]['mean'] = np.mean(results[term], axis=0)
        final_results[term]['std'] = np.std(results[term], axis=0)
    for _class in classes:
        final_results[_class] = {}
        for term in ['recall', 'precision']:
            final_results[_class][term] = {}
            final_results[_class][term]['mean'] = np.mean(results[_class][term], axis=0)
            final_results[_class][term]['std'] = np.std(results[_class][term], axis=0)
    return final_results



# Prediction gaps        
def prediction_gap_with_dataloader(model, loader, transform, gap, baseline, studied_class, indices, y_true=None, y_pred=None):
    """
    Return the average of the prediction gaps (PGs) computed on the correctly classified examples belonging to `studied_class`.

    To compute a PG for an example of a given class, one need to measure how the probability associated with the class varies when the features of the example are masked.

    Formally, let us call the probability attributed by the `model` to the class of the example "model(example)". PG is the area under the curve showing maximum(0, model(example) - model(modified example)) as a function of the number of masked features. A feature is masked when its value is replaced by the `baseline` value.

    In practice, the area is estimated with the rectangle rule: the first point is computed after masking `gap` features, the second one after masking 2`gap` features... The estimated area is the sum of the areas of the rectangles obtained (`gap` * y_value). 
    The last point is necessary the one obtained when all features are masked. If needed, a last rectangle of shape (n_feat % `gap` * y_value) is added to the estimated area. 
    The areas are divided by n_feat and by model(example).

    The biggest PG is 1 (masking one or more features leads to the maximal perturbation). 
    The smallest prediction gap is 0 (all features can be masked without perturbing the prediction). 

    Parameters:
        model  --  Neural network.
        loader  --  Dataloader.
        transform  --  Function applied to the input tensors to scale the data.
        gap  --  Area computed with the rectangle rule using points computed iteratively by masking `gap` more features each time.
        baseline  --  Values used to mask the features. Tensor (1, n_feat).
        studied_class  --  Compute the PGs for all examples belonging to a class listed in `studied_class`. List of integers.
        indices  --  Order of features to be masked. Array (n_sample, n_feat) or array (1, n_feat) when the same order is used for all examples.
        y_true  --  Classes of the examples used to generate the indices. Used for a sanity check. Tensor (n_sample, 1) or None.  
        y_pred  --  Classes predicted by `model` of the examples used to generate the indices. Used for a sanity check. Tensor (n_sample, 1) or None.
    """
    # Informations
    assert len(studied_class) > 0, "Provide a list of classes to consider for computing the PGs."
    n_feat = baseline.shape[1]
    n_class = model.variables["nb_classes"]
    device = model.fc.weight.device
    n_point = int(n_feat / gap)

    # Store the gaps
    PG = {}
    n_per_class = {}
    for c in studied_class:
        PG[c] = 0
        n_per_class[c] = 0
    
    # Compute the gaps
    torch.manual_seed(1)  # Seed needed to load the examples in the order used to generate `indices`.
    count = 0
    for i, (x, y) in enumerate(loader):
        print(i, end='\r')

        # Data
        x = x[sum(y == c for c in studied_class).bool()]
        y = y[sum(y == c for c in studied_class).bool()]
        batch_size = x.shape[0]
        x = x.to(device)
        y = y.numpy()
        if transform:
            x = transform(x)
        _class = ((model(x) > 0.5).reshape(-1) * 1.).cpu().numpy() if n_class == 2 else torch.argmax(model(x), axis=1).cpu().numpy() * 1.

        # Sanity checks
        ## Check data order with true classes
        if y_true is not None:
            assert (y == y_true[count:count + batch_size]).all(), 'Problem with data order.'
        ## Check model
        if y_pred is not None:
            assert (_class == y_pred[count:count + batch_size]).all(), 'Problem with model.'

        # Prediction gap estimated on n_point
        pred_full = model(x).reshape(-1).detach().cpu().numpy() if n_class == 2 else model(x)[np.arange(batch_size), y].detach().cpu().numpy()
        pred_gap = np.zeros((batch_size, n_point))

        for i in range(n_point):
            s = np.repeat(np.arange(batch_size), gap)
            if len(indices) == 1:
                f = np.tile(indices[0, i * gap:i * gap + gap], batch_size)
            else:
                f = indices[count:count + batch_size][:, i * gap:i * gap + gap].reshape(-1)
            x[s, f] = baseline[np.repeat(np.zeros(batch_size), gap), f].clone().detach().to(device)
            pred = model(x)
            pred = pred.reshape(-1).detach().cpu().numpy() if n_class == 2 else pred[np.arange(batch_size), y].detach().cpu().numpy()
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):  # do not print warning related to division by 0
                pred_gap[:, i] = (pred_full - pred) / pred_full
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            mask = (pred_gap) > 0 * 1.0
            pred_gap = np.sum(mask * pred_gap, axis=1) * gap / n_feat

        # Add a last point if not all features have been masked yet
        if n_feat % gap != 0:
            pred = model(baseline)[np.zeros(batch_size), 0].detach().cpu().numpy() if n_class == 2 else model(baseline)[np.zeros(batch_size), y].detach().cpu().numpy()
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                pred_gap_last_point = (pred_full - pred) / pred_full
                mask = (pred_gap_last_point) > 0 * 1.0
                pred_gap_last_point = mask * pred_gap_last_point * (n_feat % gap) / n_feat
            pred_gap += pred_gap_last_point

        # Store the PGs computed on the correctly classified examples corresponding to the studied classes
        for s in range(batch_size):
            if _class[s] == y[s]:
                if y[s] in studied_class:
                    PG[y[s]] += pred_gap[s]
                    n_per_class[y[s]] += 1

        # Update count
        count += batch_size

    # Average PG per class
    for c in studied_class:
        PG[c] = PG[c] / n_per_class[c]

    return PG



def prediction_gap_for_an_example(model, x, y, n_class, transform, gap, baseline, indices):
    """
    Return the prediction gap (PG) computed on an example `x` of a given class `y`. 

    Formally, let us call the probability attributed by the `model` to the class of the example "model(example)". PG is the area under the curve showing maximum(0, model(example) - model(modified example)) as a function of the number of masked features. A feature is masked when its value is replaced by the `baseline` value.
    
    In practice, the area is estimated with the rectangle rule: the first point is computed after masking `gap` features, the second one after masking 2`gap` features... The estimated area is the sum of the areas of the rectangles obtained (`gap` * y_value). 
    The last point is necessary the one obtained when all features are masked. If needed, a last rectangle of shape (n_feat % `gap` * y_value) is added to the estimated area.
    The area is divided by n_feat and by model(example).

    The biggest PG is 1 (masking one or more features leads to the maximal perturbation). 
    The smallest prediction gap is 0 (all features can be masked without perturbing the prediction). 

    Parameters:
        model  --  Torch model or sklearn model.
        x  --  Example (1, n_feat). Tensor (torch) or array (sklearn).
        y  --  Class of `x`. Integer.
        n_class --  Number of classes.
        transform  --  Function scaling the input `x`. Only used with a torch model.
        gap  --  Area computed with the rectangle rule using points computed iteratively by masking `gap` more features each time.
        baseline  --  Values used to mask the features (1, n_feat). Tensor (torch) or array (sklearn). 
        indices  --  Order of features to be masked. Array (1, n_feat).
    """
    # Informations
    n_feat = x.shape[1]
    model_type = str(type(model)).split('.')[0][8:]
    print(model_type)

    if model_type != "sklearn":
        device = model.fc.weight.device
        x = x.to(device)
        if transform:
            x = transform(x)
            
    # Prediction gap estimated on n_point
    n_point = int(n_feat / gap)
    if model_type == "sklearn":
        pred_full = model.predict_proba(x)[0, y]
    else:
        pred_full = model(x).reshape(-1).detach().cpu().numpy() if n_class == 2 else model(x)[0, y].detach().cpu().numpy()
    pred_gap = np.zeros((n_point))
    for i in range(n_point):
        s = np.repeat(np.arange(1), gap)
        f = indices[0, i * gap:i * gap + gap]
        x[s, f] = baseline[s, f].copy() if model_type == "sklearn" else baseline[s, f].clone().detach().to(device)
        if model_type == "sklearn":
            pred = model.predict_proba(x)[0, y]
        else:
            pred = model(x).reshape(-1).detach().cpu().numpy() if n_class == 2 else model(x)[0, y].detach().cpu().numpy()
        pred_gap[i] = (pred_full - pred) / pred_full
    mask = (pred_gap) > 0 * 1.0
    curve = list(mask * pred_gap)
    PG = np.sum(curve) * gap / n_feat

    # Add a last point if not all features have been masked yet
    if n_feat % gap != 0:
        if model_type == "sklearn":
            pred = model.predict_proba(baseline)[0, y]
        else:
            pred = model(baseline).reshape(-1).detach().cpu().numpy() if n_class == 2 else model(baseline)[0, y].detach().cpu().numpy()
        pred_gap_last_point = (pred_full - pred) / pred_full
        mask = (pred_gap_last_point) > 0 * 1.0
        curve.append(mask * pred_gap_last_point)
        PG += (mask * pred_gap_last_point) * (n_feat % gap) / n_feat

    return PG, curve



def get_features_order(attr, _type="increasing"):
    """Return an array containing the indices of the n_feat features of `attr` ranked in a certain order. The order is either
    computed from the sum of the values of the n_sample examples ("sum_increasing" or "sum_decreasing"), from the sum of the rank of
    the values of the features in each example ("rank_increasing" or "rank_decrasing"), or from the median of the rank of these
    values ("rank-median_increasing" or "rank-median_decreasing"). The rank can also be "random".

    Parameters:
        attr  --  Array of size (n_sample, n_feat).
        _type  --  Str, "random", "sum_increasing", "sum_decreasing", "rank_increasing", "rank_decreasing", "rank-median_increasing", "rank-median_decreasing".
    """
    assert _type in ["random", "sum_increasing", "sum_decreasing", "rank_increasing", "rank_decreasing", "rank-median_increasing", "rank-median_decreasing"], '_type should be in ["random", "sum_increasing", "sum_decreasing", "rank_increasing", "rank_decreasing", "rank-median_increasing", "rank-median_decreasing"]' 
    if _type == "random":
        order = np.arange(0, attr.shape[1])
        np.random.shuffle(order)
    else:
        if _type.split('_')[0] == "rank":
            ranks = np.argsort(attr, axis=1)
            values = np.sum(ranks, axis=0)
        elif _type.split('_')[0] == "rank-median":
            ranks = np.argsort(attr, axis=1)
            values = np.median(ranks, axis=0)
        elif _type.split('_')[0] == "sum":
            values = np.sum(attr, axis=0)
        if _type.split('_')[1] == "increasing":
            order = np.argsort(values)
        elif _type.split('_')[1] == 'decreasing':
            order = np.argsort(-values)
    return order.reshape(1, -1)


def from_classes_to_subclasses(data_path, name, set_name, n_subclass, labels, studied_class, other_class):
    """
    Convert all class labels appearing in labels, studied_class, other_class to subclass labels. 
    """
    # Mapping used for 'SimuB', 'SimuC' and syn_g_...
    assert name in ['SimuB', 'SimuC'] or name[:5] == 'syn_g', "Warning! Adapt the function `from_classes_to_subclasses` to your new dataset."
    mapping = {
        0: list(np.arange(n_subclass - 1)),
        1: [n_subclass - 1]}
    
    # Map studied_class
    new_studied_class = []
    for c in studied_class:
        new_studied_class += mapping[c]
    
    # Map other_class
    new_other_class = {}
    for c in other_class.keys():
        for new_c in mapping[c]:
            new_other_class[new_c] = []
            for o_c in other_class[c]:
                new_other_class[new_c] += mapping[o_c]
                
    # Map the labels of the examples
    ## Load all examples in the same order to replace the labels
    train_loader, test_loader, _, _, _, _, _, _ = load_dataloader(data_path, name, 'cpu', regroup=False)
    dataloader = train_loader if set_name=='train' else test_loader
    new_labels = []
    torch.manual_seed(1)
    for _, target in dataloader:
        target = target[torch.isin(target, torch.tensor(new_studied_class))]  # Attributions are only computed for examples whose class belongs to `studied_class`.
        size = target.shape[0]
        if size != 0:
            new_labels += list(target.cpu().detach().numpy())
    new_labels = np.array(new_labels)
    
    ## Assert that the order of the examples is correct.
    for i, c in enumerate(labels):
        assert new_labels[i] in mapping[c]
    
    return new_labels, new_studied_class, new_other_class


def get_informative_variables(studied_class, other_class, useful_group, useful_variable):
    """
    Return a list of variables enabling to identify each class listed in studied_class from the classes listed in other_classes.
    
    Parameters:
        studied_class  --  list of integers representing classes
        other_class  --  dict, associate each class (keys, e.g. 0) with a list of integers (values, e.g. [1, 2])
        useful_group  --  dict, associate each class (keys, e.g. "C0") with over-expressed groups (values)
        useful_variable  --  dict, associate groups (keys) with expressed variables (values)
    """
    counts = {}
    variables = {}
    for c in studied_class:
        variables["C"+str(c)] = []
        counts[c] = 0
        for P in useful_group["C"+str(c)]:
            for g in useful_variable[P]:
                if g not in variables["C"+str(c)]:
                    variables["C"+str(c)].append(g)
                    counts[c] += 1
        for other_c in other_class[c]:
            for P in useful_group["C"+str(other_c)]:
                for g in useful_variable[P]:
                    if g not in variables["C"+str(c)]:
                        variables["C"+str(c)].append(g)
                        counts[c] += 1
    return counts, variables


# Prediction gaps for a sklearn model      
def prediction_gap_with_dataset(model, X, y, gap, baseline, studied_class, indices):
    """
    Return the average of the prediction gaps (PGs) computed on the correctly classified examples belonging to `studied_class`.

    To compute a PG for an example of a given class, one need to measure how the probability associated with the class varies when the features of the example are masked.

    Formally, let us call the probability attributed by the `model` to the class of the example "model(example)". PG is the area under the curve showing maximum(0, model(example) - model(modified example)) as a function of the number of masked features. A feature is masked when its value is replaced by the `baseline` value.

    In practice, the area is estimated with the rectangle rule: the first point is computed after masking `gap` features, the second one after masking 2`gap` features... The estimated area is the sum of the areas of the rectangles obtained (`gap` * y_value). 
    The last point is necessary the one obtained when all features are masked. If needed, a last rectangle of shape (n_feat % `gap` * y_value) is added to the estimated area. 
    The areas are divided by n_feat and by model(example).

    The biggest PG is 1 (masking one or more features leads to the maximal perturbation). 
    The smallest prediction gap is 0 (all features can be masked without perturbing the prediction). 

    Parameters:
        model  --  Sklearn classifier.
        X  --  Data. Array (n_sample, n_feat).
        y  --  Labels. Array (n_sample,).
        gap  --  Area computed with the rectangle rule using points computed iteratively by masking `gap` more features each time.
        baseline  --  Values used to mask the features. Array (1, n_feat).
        studied_class  --  Compute the PGs for all examples belonging to a class listed in `studied_class`. List of integers.
        indices  --  Order of features to be masked. Array (1, n_feat).
    """
    # Informations
    assert len(studied_class) > 0, "Provide a list of classes to consider for computing the PGs."
    n_feat = X.shape[1]
    n_point = int(n_feat / gap)

    # Store the gaps
    PG = {}
    n_per_class = {}
    for c in studied_class:
        PG[c] = 0
        n_per_class[c] = 0
    
    # Data
    X = X[sum(y == c for c in studied_class).astype(bool)]
    y = y[sum(y == c for c in studied_class).astype(bool)]
    n_sample = X.shape[0]
    _class = model.predict(X)

    # Prediction gap estimated on n_point
    pred_full = model.predict_proba(X)[np.arange(n_sample), y]
    pred_gap = np.zeros((n_sample, n_point))

    for i in range(n_point):
        s = np.repeat(np.arange(n_sample), gap)  # [0 0 0 1 1 1 2 ..]
        f = np.tile(indices[0, i * gap:i * gap + gap], n_sample)  # [0 1 2 0 1 2 ..]
        X[s, f] = baseline[np.repeat(np.zeros(n_sample, dtype=int), gap), f].copy()
        pred = model.predict_proba(X)[np.arange(n_sample), y]
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):  # do not print warning related to division by 0
            pred_gap[:, i] = (pred_full - pred) / pred_full
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        mask = (pred_gap) > 0 * 1.0
        pred_gap = np.sum(mask * pred_gap, axis=1) * gap / n_feat

    # Add a last point if not all features have been masked
    if n_feat % gap != 0:
        pred = model.predict_proba(baseline)[np.zeros(n_sample, dtype=int), y]
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            pred_gap_last_point = (pred_full - pred) / pred_full
            mask = (pred_gap_last_point) > 0 * 1.0
            pred_gap_last_point = mask * pred_gap_last_point * (n_feat % gap) / n_feat
        pred_gap += pred_gap_last_point

    # Store the PGs computed on the correctly classified examples corresponding to the studied classes
    for s in range(n_sample):
        if _class[s] == y[s]:
            if y[s] in studied_class:
                PG[y[s]] += pred_gap[s]
                n_per_class[y[s]] += 1

    # Average PG per class
    for c in studied_class:
        PG[c] = PG[c] / n_per_class[c]

    return PG



