# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import torch
from sklearn.metrics import balanced_accuracy_score
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


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, DiffuseLR, DiffuseMLP)")
argParser.add_argument("--exp", type=int, help="experiment number", default=1)
args = argParser.parse_args()
name = args.name
model_name = args.model
exp = args.exp
print('Model    ', model_name)


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Dataset
train_loader, test_loader, n_class, n_feat, class_name, feat_name, transform, n_sample = load_dataloader(data_path, name, device)
print(f"In our dataset, we have {n_class} classes and {n_sample} examples. Each example contains {n_feat} features.")


# Model
softmax = False
n_layer, n_hidden_feat = get_hyperparameters(name, model_name)
model = load_model(model_name, n_feat, n_class, softmax, device, save_path, n_layer, n_hidden_feat)
save_name = os.path.join(model_name, f"exp_{exp}")


# Optimization
criterion, optimizer, scheduler, n_epoch = set_optimizer(name, model)


# Train
epochs_acc = []
epochs_loss = []
for epoch in range(n_epoch):
    epoch_loss, epoch_acc = train(model, criterion, optimizer, train_loader, device, transform)  # train for 1 epoch
    print("\rLoss at epoch {}: {:.2f}.".format(epoch+1, epoch_loss), end='')
    print("(Acc \t: {:.2f}).".format(epoch_acc*100),end='')
    scheduler.step()
## Score
y_pred, y_true = predict(model, train_loader, device, transform)
train_score = compute_accuracy_from_predictions(y_pred, y_true)
print(f'The training accuracy with our {model.name} is {np.round(train_score, 2)}.')
train_balanced_score = balanced_accuracy_score(y_true, y_pred) * 100
print(f'The balanced training accuracy with our {model.name} is {np.round(train_balanced_score, 2)}.')

# Test
## Score
y_pred, y_true = predict(model, test_loader, device, transform)
test_score = compute_accuracy_from_predictions(y_pred, y_true)
print(f'The test accuracy with our {model.name} is {np.round(test_score, 2)}.')
test_balanced_score = balanced_accuracy_score(y_true, y_pred) * 100
print(f'The balanced test accuracy with our {model.name} is {np.round(test_balanced_score, 2)}.')
correct_test_indices = np.argwhere(y_pred == y_true)
## Confusion matrix
cm = get_confusion_matrix(y_true, y_pred, class_name, normalize='true')
create_new_folder(os.path.join(save_path, save_name))
plot_confusion_matrix(cm, file=os.path.join(save_path, save_name, "confusion_matrix.png"), show=False)


# Save
with open(os.path.join(save_path, save_name, "accuracy.csv"), "w") as f:
    f.write(f"train, {np.round(train_score, 2)}\n")
    f.write(f"balanced_train, {np.round(train_balanced_score, 2)}\n")
    f.write(f"test, {np.round(test_score, 2)}\n")
    f.write(f"balanced_test, {np.round(test_balanced_score, 2)}\n")

torch.save({'epoch': epoch+1,
            'arch': "{}".format(model.name),
            'variables': "{}".format(model.variables),
            'state_dict': model.state_dict(),
            'train_acc': train_score,
            'test_acc': test_score,
            'correct_test_indices': correct_test_indices,
            }, os.path.join(save_path, save_name, "checkpoint.pt"))
