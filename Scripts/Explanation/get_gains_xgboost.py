# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
import random
from collections import OrderedDict
from joblib import load
import argparse
from setting import *
from utils import *
from loader import *


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("--exp", type=int, help="experiment number", default=1)
args = argParser.parse_args()
name = args.name
model_name = "xgboost"
exp = args.exp


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)
save_name = os.path.join(model_name, f"exp_{exp}")


# Seed
random.seed(exp)
np.random.seed(exp)


# Dataset
_, _, _, _, n_class, n_feat, class_name, feat_name = load_dataset(data_path, name, normalize=False)
feat_name = np.array(feat_name)

 
# Model
model = load(os.path.join(save_path, save_name, "checkpoint.joblib"))


# Explanation (using gain)
booster = model.get_booster()
importance = booster.get_score(importance_type='gain')
print("Number of genes used by the model: ", len(importance.keys()))
scores = [importance.get(f"f{i}", 0) for i in range(n_feat)]
scores = np.array(scores)

# Save
order = np.argsort(-scores)
np.save(os.path.join(save_path, "order", f"order_xgboost_exp_{exp}.npy"), feat_name[order])
np.save(os.path.join(save_path, "order", f"order_xgboost_exp_{exp}_values.npy"), scores[order])

