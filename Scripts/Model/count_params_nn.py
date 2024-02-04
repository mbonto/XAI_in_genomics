# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import torch
import argparse
from setting import *
from loader import *
from models import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, GNN)")
argParser.add_argument("--n_repet", type=int, help="Results are averaged for all experiments between 1 and `n_repet`")
argParser.add_argument("--selection", type=str, help="method used to select features (var, PCA_PC1, F, MI, L1_exp_1, DESeq2, IG_LR_set_train_exp_1, IG_MLP_set_train_exp_1, IG_GCN_set_train_exp_1)")
argParser.add_argument("--n_feat_selected", type=int, help="number of features selected.")
argParser.add_argument("--selection_type", type=str, choices=["best", "worst", "random_wo_best"], help="when `selection` is given, keep best, worst or random without best features.")
args = argParser.parse_args()
name = args.name
model_name = args.model
n_repet = args.n_repet
selection = args.selection
n_feat_selected = args.n_feat_selected
selection_type = args.selection_type
print('Model    ', model_name)


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Seed
exps = np.arange(1, n_repet + 1)
params = 0
for exp in exps:
    seed = exp if selection is None and n_feat_selected is None else exp + 100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    
    # Information
    studied_features = get_selected_features(selection, selection_type, n_feat_selected, save_path)
    _, _, n_class, n_feat, _, _, _, _ = load_dataloader(data_path, name, device, studied_features=studied_features)
    
    
    # Model
    softmax = False
    n_layer, n_hidden_feat, graph_name = get_hyperparameters(name, model_name)
    model = load_model(model_name, n_feat, n_class, softmax, device, save_path, n_layer, n_hidden_feat, graph_name, studied_features)
    n_param = 0
    for param in model.parameters():
        temp = 1
        for i in param.shape:
            temp = temp * i
        n_param += temp
    print(n_param)
    params += n_param
print("final    ", params / n_repet)

