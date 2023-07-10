# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
import argparse
from sklearn.metrics import jaccard_score
from setting import *
from loader import *
from XAI_method import *
from XAI_interpret import *
from utils import create_new_folder, set_pyplot


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, DiffuseLR, DiffuseMLP)")
argParser.add_argument("--set", type=str, help="set (train, test)")
argParser.add_argument("--n_repet", type=int, help="Results are averaged for all experiments between 1 and `n_repet`")
args = argParser.parse_args()
name = args.name
model_name = args.model
set_name = args.set
n_repet = args.n_repet
print('Dataset    ', name)
print('Model    ', model_name)
XAI_method = "Integrated_Gradients"
method = "mean"


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Load the attributions of all the examples of the set
attr_exps = {}
arg_exps = {}
exps = np.arange(1, n_repet + 1)

for exp in exps:
    save_name = os.path.join(model_name, f"exp_{exp}", XAI_method)
    attr_exp = np.load(os.path.join(save_path, save_name, "{}_scores_with_{}_{}.npy".format(XAI_method, method, set_name)), allow_pickle=True).item()
    attr_exps[exp] = attr_exp
    arg_exp = {}
    for key in attr_exp.keys():
        if key != "general":
            # print("key", key)
            attr_exp_class = attr_exp[key]['attr']
            arg_exp_class = np.argsort(-attr_exp_class)
            # print("arg_exp_class", arg_exp_class.shape)
            arg_exp[key] = arg_exp_class
    arg_exps[exp] = arg_exp


# Calculate the Jaccard scores between all the models
avg_jaccard_scores  = []
n_args = [1, 5, 10, 100, 500, 1000]

for n in n_args:
    jaccard_scores = []
    for exp1 in exps:
        for exp2 in range(exp1 + 1, n_repet + 1):
            if n == 100:
                print(exp1, exp2)
            scores = []
            for _class in arg_exps[exp1].keys():
                # print("class", _class)
                arg1 = set(arg_exps[exp1][_class][:n])
                arg2 = set(arg_exps[exp2][_class][:n])
                s = len(list(arg1.intersection(arg2))) / n # / len(list(arg1.union(arg2)))
                if n == 100:
                    print(s)
                scores.append(s)
            jaccard_scores.append(np.mean(scores))
    avg_jaccard_scores.append(np.mean(jaccard_scores))


# Save
create_new_folder(os.path.join(save_path, "figures"))
np.savez(os.path.join(save_path, "figures", "Jaccard_scores_with_{}_{}_on_{}".format(model_name, XAI_method, set_name)), n_args, avg_jaccard_scores)


# Plot
set_pyplot()
plt.figure(figsize=(15, 10))
plt.plot(n_args, avg_jaccard_scores, 'x-')
plt.xlabel("Number of variables")
plt.ylabel(f"Average Jaccard score within a {method}")
plt.savefig(os.path.join(save_path, "figures", "Jaccard_scores_with_{}_{}_on_{}".format(model_name, XAI_method, set_name)), bbox_inches='tight', dpi=150)


