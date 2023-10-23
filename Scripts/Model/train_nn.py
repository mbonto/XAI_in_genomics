# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import torch
from sklearn.metrics import balanced_accuracy_score
import time
import argparse
from setting import *
from utils import *
from dataset import *
from loader import *
from plots_and_stats import *
from evaluate import *
from models import *
from training import *
set_pyplot()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, GNN)")
argParser.add_argument("--exp", type=int, help="experiment number", default=1)
argParser.add_argument("--selection", type=str, help="method used to select features (var, PCA_PC1, F, MI, L1_exp_1, DESeq2, IG_LR_set_train_exp_1, IG_MLP_set_train_exp_1, IG_GCN_set_train_exp_1)")
argParser.add_argument("--n_feat_selected", type=int, help="number of features selected.")
argParser.add_argument("--selection_type", type=str, choices=["best", "worst", "random_wo_best"], help="when `selection` is given, keep best, worst or random without best features.")
args = argParser.parse_args()
name = args.name
model_name = args.model
exp = args.exp
selection = args.selection
n_feat_selected = args.n_feat_selected
selection_type = args.selection_type
print('Model    ', model_name)


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)
save_name = os.path.join(model_name, f"exp_{exp}") if selection is None and n_feat_selected is None else os.path.join(model_name, "FS")


# Seed
seed = exp if selection is None and n_feat_selected is None else exp + 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device == 'cuda':
    torch.cuda.manual_seed_all(seed)


# Dataset
studied_features = get_selected_features(selection, selection_type, n_feat_selected, save_path)
train_loader, test_loader, n_class, n_feat, class_name, feat_name, transform, n_sample = load_dataloader(data_path, name, device, studied_features=studied_features)
if selection is None and n_feat_selected is None:
    print(f"In our dataset, we have {n_class} classes and {n_sample} examples. Each example contains {n_feat} features.")


# Model
softmax = False
n_layer, n_hidden_feat, graph_name = get_hyperparameters(name, model_name)
model = load_model(model_name, n_feat, n_class, softmax, device, save_path, n_layer, n_hidden_feat, graph_name, studied_features)
# print(model)
# for param in model.parameters():
#     print(param.shape)


# Optimization
criterion, optimizer, scheduler, n_epoch = set_optimizer(name, model)


# Train
start_time = time.time()
epochs_acc = []
epochs_loss = []
for epoch in range(n_epoch):
    epoch_loss, epoch_acc = train(model, criterion, optimizer, train_loader, device, transform)  # train for 1 epoch
    print("\rLoss at epoch {}: {:.2f}. (Acc: {:.2f})".format(epoch+1, epoch_loss, epoch_acc*100), end='')
    scheduler.step()
duration = time.time() - start_time
## Score
y_pred, y_true = predict(model, train_loader, device, transform)
train_score = compute_accuracy_from_predictions(y_pred, y_true)
print("")
if selection is None and n_feat_selected is None:
    print(f'The training accuracy with our {model.name} is {np.round(train_score, 2)}.') 
train_balanced_score = balanced_accuracy_score(y_true, y_pred) * 100
print(f'The balanced training accuracy with our {model.name} is {np.round(train_balanced_score, 2)}.')


# Test
## Score
y_pred, y_true = predict(model, test_loader, device, transform)
test_score = compute_accuracy_from_predictions(y_pred, y_true)
if selection is None and n_feat_selected is None:
    print(f'The test accuracy with our {model.name} is {np.round(test_score, 2)}.')
test_balanced_score = balanced_accuracy_score(y_true, y_pred) * 100
print(f'The balanced test accuracy with our {model.name} is {np.round(test_balanced_score, 2)}.')
correct_test_indices = np.argwhere(y_pred == y_true)
## Confusion matrix
create_new_folder(os.path.join(save_path, save_name))
if selection is None and n_feat_selected is None:
    cm = get_confusion_matrix(y_true, y_pred, class_name, normalize='true')
    plot_confusion_matrix(cm, file=os.path.join(save_path, save_name, "confusion_matrix.png"), show=False)


# Save
file_name = "accuracy.csv" if selection is None and n_feat_selected is None else f"{exp}_{selection}_{n_feat_selected}_{selection_type}.csv"
with open(os.path.join(save_path, save_name, file_name), "w") as f:
    f.write(f"train, {np.round(train_score, 2)}\n")
    f.write(f"balanced_train, {np.round(train_balanced_score, 2)}\n")
    f.write(f"test, {np.round(test_score, 2)}\n")
    f.write(f"balanced_test, {np.round(test_balanced_score, 2)}\n")
    f.write(f"duration, {np.round(duration, 2)}\n")

if selection is None and n_feat_selected is None:
    print("Model saved.")
    torch.save({'epoch': epoch+1,
                'arch': "{}".format(model.name),
                'variables': "{}".format(model.variables),
                'state_dict': model.state_dict(),
                'train_acc': train_score,
                'test_acc': test_score,
                'correct_test_indices': correct_test_indices,
                }, os.path.join(save_path, save_name, "checkpoint.pt"))

