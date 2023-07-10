import torch
from captum.attr import IntegratedGradients, KernelShap
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
    

def compute_attributes_from_a_dataloader(model, dataloader, transform, device, studied_class, method="Integrated_Gradients", n_steps=100, n_samples=10, baseline=None):
    """
    Return importance scores attributed by an XAI method to each feature of each input.
    
    Parameters:
        model  --  Neural network.
        dataloader  --  Dataloader.
        transform  --  Function applied on input tensors to scale data.
        device  --  `cpu` or `cuda`.
        studied_class  --  Compute the scores for all examples belonging to a class in `studied_class`. List of integers.
        method  --  "Integrated_Gradients", "Kernel_Shap".
        n_steps  --  Used if method == "Integrated_Gradients". Integer.
        n_samples  --  Used if method == "Kernel_Shap". Integer.
        baseline  --  Input used as a reference for "Integrated_Gradients". Tensor (1, n_feat).
    """
    # Informations
    assert len(studied_class) > 0, "Provide a list of classes to consider for computing the attributions."
    x, _ =  next(iter(dataloader))
    n_feat = x.shape[1]
    n_sample = 0
    for x, y in dataloader:
        n_sample += torch.sum(sum(y == c for c in studied_class)).item()
    
    # Compute the attributions
    attr = torch.zeros(n_sample, n_feat).to(device)
    y_pred = np.ones(n_sample)
    y_true = np.ones(n_sample, dtype='int')
    if method == "Integrated_Gradients":
        xai = IntegratedGradients(model)
    elif method == "Kernel_Shap":
        xai = KernelShap(model)   
    torch.manual_seed(1)
    count = 0
    for i, (x, target) in enumerate(dataloader):
        print(i, end='\r')
        x = x[sum(target == c for c in studied_class).bool()]
        target = target[sum(target == c for c in studied_class).bool()]
        batch_size = x.shape[0]
        x = x.to(device)
        if transform:
            x = transform(x)
        target = target.to(device)
        
        if method == "Integrated_Gradients":
            # For each input, the sum of the attributions should be equal to model(input) - model(baseline).
            # If n_step is too small, this statement might not be true due to approximation errors.
            # Here, we accept any absolute difference lower than 1 (considering that the highest possible probability is 100).
            # If the gap is bigger, the attributions are computed by an increased number of steps.
            valid = False
            add = 0
            while not valid:
                attrs = xai.attribute(x, target=target, n_steps=n_steps+add, baselines=baseline, internal_batch_size=batch_size)
                gap = check_ig_from_array(attrs.cpu().detach().numpy(), model, x, target, baseline)
                if gap > 1:
                    add += n_steps
                    print(f"Maximal gap: {np.round(gap, 6)}. There is at least one sample for which the difference between the sum of the attributions and the predictions is higher than 1. Run again IG with {n_steps+add} steps.")
                else:
                    valid = True
            attr[count:count + batch_size, :] = attrs
        elif method == "Kernel_Shap":
            attr[count:count + batch_size, :] = xai.attribute(x, target=target, n_samples=n_samples)
        outputs = model(x)
        _, pred = torch.max(outputs.data, 1)
        
        y_true[count:count + batch_size] = target.cpu().detach().numpy()
        y_pred[count:count + batch_size] = pred.cpu().detach().numpy()
        count = count + batch_size
    attr = attr.detach().cpu().numpy()
    return attr, y_true, y_pred


def check_ig_from_a_dataloader(attr, model, dataloader, transform, device, baseline, studied_class, save_name=None, show=True):
    """
    For each input, we should have `sum of the attributions = model(input) - model(baseline)`.
    If this is not the case, increase the number of steps and recompute the attributions.
    
    Parameters:
        attr  --  Attributions. Tensor (n_sample, n_feat).
        model  --  Neural network.
        dataloader  --  Dataloader.
        transform  --  Function applied on input tensors to scale data.
        device  --  `cpu` or `cuda`.
        baseline  --  Input used as a reference for "Integrated_Gradients". Tensor (1, n_feat).
        save_name  --  Path used to store the figure.
        studied_class  --  Compute the scores for all examples belonging to a class in `studied_class`. List of integers.
        show  --  Show the plot or not. True or False.
    """
    _sum = np.round(np.sum(attr, axis=1) * 100, decimals=2)
    n_sample = len(attr)
    output_X = np.zeros(n_sample)
    output_baseline = np.zeros(n_sample)
    
    torch.manual_seed(1)
    count = 0
    for i, (x, target) in enumerate(dataloader):
        print(i, end='\r')
        x = x[sum(target == c for c in studied_class).bool()]
        target = target[sum(target == c for c in studied_class).bool()]
            
        batch_size = x.shape[0]
        
        x = x.to(device)
        
        if transform:
            x = transform(x)
            
        target = target.to(device)
        
        output_X[count:count + batch_size] = torch.take_along_dim(model(x), target.reshape(-1, 1), dim=1).reshape(-1).cpu().detach().numpy()
        output_baseline[count:count + batch_size] = torch.take_along_dim(model(baseline).repeat(x.shape[0], 1), target.reshape(-1, 1), dim=1).reshape(-1).cpu().detach().numpy()
        count = count + batch_size
    
    diff = np.round((output_X - output_baseline) * 100, 2)
    
    # Results
    sns.violinplot(x=_sum-diff)
    plt.xlabel("sum of the attributions - (model(input) - model(baseline))", labelpad=50)
        
    if save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=150)
    
    if show:
        plt.show()
    plt.close('all')
    
    return np.max(np.abs(_sum-diff))
 

def check_ig_from_array(attr, model, x, target, baseline):
    """
    For each input, we should have `sum of the attributions = model(input) - model(baseline)`.
    If this is not the case, increase the number of steps and recompute the attributions.
    
    Parameters:
    attr  --  Attributions, tensor (n_sample, n_feat).
    model  -- Neural network.
    x  --  Inputs, tensor (n_sample, n_feat).
    target  --  Labels, tensor (n_sample).
    baseline  -- Reference input, tensor (1, n_feat).
    """
    _sum = np.round(np.sum(attr, axis=1) * 100, decimals=2)

    output_X = torch.take_along_dim(model(x), target.reshape(-1, 1), dim=1).reshape(-1).cpu().detach().numpy()
    output_baseline = torch.take_along_dim(model(baseline).repeat(x.shape[0], 1), target.reshape(-1, 1), dim=1).reshape(-1).cpu().detach().numpy()
    diff = np.round((output_X - output_baseline) * 100, 2)
    
    return np.max(np.abs(_sum-diff))


def get_baseline(train_loader, device, n_feat, transform, base_class=None):
    if base_class is not None:
        baseline = torch.zeros(1, n_feat).to(device)
        count = 0
        for x, y in train_loader:
            x = x.to(device)
            if transform is not None:
                x = transform(x)
            baseline += torch.sum(x[y==base_class], axis=0).reshape(1, -1)
            count += len(x[y==base_class])
        baseline = baseline / count
    else:
        baseline = torch.zeros(1, n_feat).to(device)
    return baseline


def load_attributions(XAI_method, save_path, set_name):
    checkpoint = torch.load(os.path.join(save_path, '{}_{}.pt'.format(XAI_method, set_name)))
    return checkpoint['features_score'], checkpoint['predictions'], checkpoint['true_labels'], checkpoint['labels_name'], checkpoint['features_name'], checkpoint['baseline'], checkpoint['baseline_pred']


def save_attributions(features_score, features_name, model, XAI_method, predictions, true_labels, baseline, baseline_pred, labels_name, save_path, set_name):
    torch.save({'features_score': features_score,
            'predictions': predictions,
            'true_labels': true_labels,
            'labels_name': labels_name,
            'features_name': features_name,
            'variables': model.variables,
            'name': model.name,
            'baseline': baseline,
            'baseline_pred': baseline_pred
            }, os.path.join(save_path, "{}_{}.pt".format(XAI_method, set_name)))
