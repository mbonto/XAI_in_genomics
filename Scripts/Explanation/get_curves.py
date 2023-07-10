# Libraries
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
import torch
import argparse
from setting import *
from utils import *
from dataset import *
from loader import *
from evaluate import *
from models import *
from XAI_method import *
from XAI_visualise import *
from XAI_interpret import *
set_pyplot()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, DiffuseLR, DiffuseMLP)")
argParser.add_argument("--set", type=str, help="set (train, test)")
argParser.add_argument("--simu", type=int, help="number of repetitions for random selection")
argParser.add_argument("--exp", type=int, help="experiment number", default=1)
args = argParser.parse_args()
name = args.name
model_name = args.model
set_name = args.set
n_simu = args.simu
exp = args.exp
print('Model    ', model_name)
XAI_method = "Integrated_Gradients"


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)
save_name = os.path.join(model_name, f"exp_{exp}")


# Dataset
train_loader, test_loader, n_class, n_feat, class_name, feat_name, transform, n_sample = load_dataloader(data_path, name, device)
if set_name == "train":
    loader = train_loader
elif set_name == "test":
    loader = test_loader


# Baseline
base_class, studied_class = get_XAI_hyperparameters(name, n_class)
baseline = get_baseline(train_loader, device, n_feat, transform, base_class)


# Model
softmax = True
n_layer, n_hidden_feat = get_hyperparameters(name, model_name)
model = load_model(model_name, n_feat, n_class, softmax, device, save_path, n_layer, n_hidden_feat)
checkpoint = torch.load(os.path.join(save_path, save_name, 'checkpoint.pt'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()
    

# Attributions
attr, y_pred, y_true, labels, features, baseline, _ = load_attributions(XAI_method, os.path.join(save_path, save_name, XAI_method), set_name=set_name)
baseline = torch.from_numpy(baseline)


# Store all examples in a tensor
X = torch.zeros(len(attr), n_feat)
Y = torch.zeros(len(attr))
torch.manual_seed(1)
count = 0
for x, y in loader:
    x = x[sum(y == c for c in studied_class).bool()]
    y = y[sum(y == c for c in studied_class).bool()]
    X[count:count+len(y)] = x
    Y[count:count+len(y)] = y
    count += len(y)
X, Y = X.to(device), Y.to(device)
if transform:
    X = transform(X)
    

# Assert that the model and data are coherent
assert compute_accuracy_from_model_with_dataloader(model, train_loader, transform, device) == checkpoint['train_acc']
assert compute_accuracy_from_model_with_dataloader(model, test_loader, transform, device) == checkpoint['test_acc']
if set_name == "train":
    assert compute_accuracy_from_model(model, X, Y.reshape(-1, 1)) == compute_accuracy_from_predictions(y_pred, y_true)
else:
    assert compute_accuracy_from_model(model, X, Y.reshape(-1, 1)) == compute_accuracy_from_predictions(y_pred, y_true)
    
    
# Keep correctly classified samples only
correct_indices = np.argwhere((y_pred - y_true) == 0)[:, 0]
print("There are {} uncorrect examples. We remove them from our study.".format(len(y_pred) - len(correct_indices)))
attr = attr[correct_indices]
y_true = y_true[correct_indices]
y_pred = y_pred[correct_indices]
Y = Y.reshape(-1).type(torch.long)[correct_indices]
X = X[correct_indices]
assert (Y.cpu().numpy() == y_true).all()

# Normalize
attr = transform_data(attr, transform='divide_by_norm')


# Attributions averaged per class
scores = np.load(os.path.join(save_path, save_name, XAI_method, "{}_scores_with_{}_{}.npy".format(XAI_method, "mean", set_name)), allow_pickle=True).item()


# Curves
classes = []  # precision and recall are computed for each class in this list

# Performance when the K most important features of each class are kept/modified
K = np.arange(1, n_feat, 1)  # np.concatenate((np.arange(1, 10, 1), np.arange(10, 200, 10), np.arange(200, n_feat, 100)))
## First, the K most important are kept.
res_cls_best, n_kept_cls_best, kept_feat_cls_best = get_results_per_class(model, X, Y, K, scores, 'keep_best', baseline, classes)
## Then, the K most important are modified. 
res_cls_worst, n_kept_cls_worst, kept_feat_cls_worst = get_results_per_class(model, X, Y, K[:-1], scores, 'remove_best', baseline, classes)

# Performance when the K most important features of each example are kept/modified
K = np.arange(1, n_feat, 1)  # np.concatenate((np.arange(1, 10, 1), np.arange(10, 100, 10), np.arange(100, n_feat, 100)))
## Average over all examples adjusted by the proportion of each class
res_bal_best, kept_feat_bal_best = get_results_with_best_features_kept_or_removed(model, X, Y, K, attr, baseline, classes, kept=True, balance=True)
res_bal_worst, kept_feat_bal_worst = get_results_with_best_features_kept_or_removed(model, X, Y, K, attr, baseline, classes, kept=False, balance=True)
## Average over all examples
res_best, kept_feat_best = get_results_with_best_features_kept_or_removed(model, X, Y, K, attr, baseline, classes, kept=True, balance=False)
res_worst, kept_feat_worst = get_results_with_best_features_kept_or_removed(model, X, Y, K, attr, baseline, classes, kept=False, balance=False)

# Performance when random genes are removed
res_rand = get_results_with_random_features(model, X, Y, K, n_simu, baseline, classes)
lim = np.argmax(np.argwhere((K <= n_feat/2)))
res_rand_wo_bal_best = get_results_with_random_features(model, X, Y, K[:lim], n_simu, baseline, classes, kept_feat_bal_best[:lim])
res_rand_wo_best = get_results_with_random_features(model,X, Y, K[:lim], n_simu, baseline, classes, kept_feat_best[:lim])
n_kept_cls_best = np.array(n_kept_cls_best)
lim_cls = np.argmax(np.argwhere((n_kept_cls_best <= n_feat/2)))
res_rand_wo_cls_best = get_results_with_random_features(model, X, Y, n_kept_cls_best[:lim_cls], n_simu, baseline, classes, kept_feat_cls_best[:lim_cls])


# Save
results = {
    'indices': K,
    'res_best': res_best,
    'res_worst': res_worst,
    'res_bal_best': res_bal_best,
    'res_bal_worst': res_bal_worst,
    'res_rand': res_rand,
    'res_rand_wo_best': res_rand_wo_best,
    'res_rand_wo_bal_best': res_rand_wo_bal_best,
    'indices_cls_best': n_kept_cls_best,
    'indices_cls_worst': n_kept_cls_worst,
    'res_cls_best': res_cls_best,
    'res_cls_worst': res_cls_worst,
    'res_rand_wo_cls_best': res_rand_wo_cls_best,
}
np.save(os.path.join(save_path, save_name, XAI_method, "figures", "Curves_on_{}_with_{}.npy".format(set_name, XAI_method)), results)


# Plot
save_file = os.path.join(save_path, save_name, XAI_method, 'figures', f'Curves_on_{set_name}_with_{XAI_method}.png')
xlabel = "Number of features kept"
ylabel = "Balanced accuracy (%)"
plot_TCGA_results(results, xlabel, ylabel, save_file, show=False)

