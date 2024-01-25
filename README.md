# XAI_in_genomics

This repository contains the code of the experiments performed in the following paper\
[A Comparative Analysis of Gene Expression Profiling
by Statistical and Machine Learning Approaches]()
by Myriam Bontonou, Anaïs Haget, Maria Boulougouri, Benjamin Audit, Pierre Borgnat, Jean-Michel Arbona.


## Abstract
Many machine learning models have been proposed to classify phenotypes from gene expression data. In addition to their good performance, these models have the potential to provide some understanding of phenotype by extracting explanations for their decisions. These explanations often take the form of a list of genes ranked in order of importance for the predictions, the best-ranked genes being interpreted as linked to the phenotype. We discuss the biological and the methodological limitations of such explanations. Experiments are performed on several datasets gathering cancer and healthy tissue samples from the TCGA, GTEx and TARGET databases. A collection of machine learning models including logistic regression, multilayer perceptron, and graph neural network are trained to classify samples according to their cancer type. Gene rankings are obtained from explainability methods adapted to these models, and compared to the ones from classical statistical feature selection methods such as mutual information, DESeq2, and edgeR. Interestingly, on simple tasks, we observe that the information learned by black-box neural networks is related to the notion of differential expression. In all cases, a small set containing the best-ranked genes is sufficient to achieve a good classification. However, the top-ranked genes differ significantly between the methods and similar classification performance can be achieved with lower ranked genes. In conclusion, although these methods enable to identify biomarkers characteristic of certain pathologies, our results question the completeness of the selected gene sets and thus of explainability by the identification of the underlying biological processes.

## Usage
### 1. Dependencies
- Python = 3.7
- PyTorch = 1.11
- PyTorch geometric = 2.0


### 2. Download datasets
The datasets will be stored in a folder on your computer. Set the absolute path of this folder in the function set_path in setting.py.

*The datasets gather data coming from the TCGA, TARGET and GTEx databases [1]. More details on the datasets are presented in the notebooks `Describe_data.ipynb` and `Discover_gene_expression_data.ipynb` contained in their respective folders.*

##### PanCan
Go to the Pancan/Data folder and execute `python get_pancan.py`.

##### BRCA
Go to the Gdc/Data folder and execute `python get_gdc.py`.

##### BRCA-pam
Go to the Legacy/Data folder and execute `python get_legacy.py`.

##### ttg-breast and ttg-all
Go to the TTG/Data folder and execute `python get_ttg.py`.


### 3. Learning models
In the following, the same commands can be used for various datasets and learning models.
- Various datasets can be used: pancan (PanCan), BRCA, BRCA-pam, ttg-breast, ttg-all. 
- Various models can be trained using PyTorch - logistic regression (LR), multilayer perceptron (MLP), graph neural network (GCN), or scikit-learn - logistic regression (LR_L1_penalty, LR_L2_penalty).

Go to Scripts/Model.
#### Graph
To compute the correlation graph over all features using training examples, execute `python infer_graph.py -n [dataset_name]`.

#### Model
To train a torch model (LR, MLP, GCN) on a dataset, execute `python train_nn.py -n [dataset_name] -m [model_name]`.

To train a scikit-learn model (LR_L1_penalty, LR_L2_penalty) on a dataset, execute `python train_sklearn.py -n [dataset_name] -m [model_name]`.


### 4. Explainability
Go to Scripts/Explanation.

To compute the integrated gradients scores on the training examples, execute `python get_attributions.py -n [dataset_name] -m [model_name] --set train`.

To compute the scores averaged over all studied classes, execute `python get_attributions_averaged_per_class.py -n [dataset_name] -m [model_name] --set train`.

To compute the prediction gaps (PGs), execute `python get_prediction_gaps.py -n [dataset_name] -m [model_name] --set train`. Local PGs are obtained by ranking the features of each example independently. Global PGs are obtained by ranking them in the same way for all examples of the same class.


## Results
### 1. Datasets
For more details, please have a look at the scientific articles.

|  Name  | # classes | # samples (min/max per class)  | # variables |
|:---------:|:-----------:|:-------------------------:|:-----------:|
| pancan    |     33      |     9680 (36/1095)        |     15401   |
|   BRCA    |     2       |     1210 (113/1097)       |     13946   |
| BRCA-pam  |     5       |     916 (67/421)          |     13896   |
|ttg-breast |      2      |     1384 (292/1092)       |     14373   |
|  ttg-all  |      2      |     17600(8130/9470)      |     14368   |


### 2. Learning
Each model is trained 10 times with a different random initialisation. The results presented here are the averages and standard deviations obtained with the 10 learned models.

Balanced accuracy (%)
| Dataset           | LR+L1             | LR+L2             | MLP               | GNN               |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| PanCan            | 95.0              |94.3               |94.3 +- 0.3        |92.1 +- 0.4        |
| BRCA              | 99.7              |98.5               |99.5 +- 0.4        |98.9 +- 0.6        |
| BRCA-pam          | 92.3              |90.7 +- 0.2        |87.4 +- 1.8        |87.1 +- 1.4        |
| ttg-breast        | 99.7              |99.2               |99.4 +- 0.3        |99.1 +- 0.1        |
| ttg-all           | 99.5              |99.5               |99.6               |99.4 +- 0.1        |



### 3. Explaining
The scores attributed to the variables are computed with integrated gradients for each example correctly classified of the training set. The importance of the value of a variable for a prediction is computed with respect to a default prediction on a reference example (called baseline). 

|  Name      |            Baseline                     |     Studied classes      | 
|:----------:|:---------------------------------------:|:------------------------:|
| pancan     | Average of the training samples         |       All                |
| BRCA       | Average of the normal training samples  | Tumour samples           |
| BRCA-pam   | Average of the normal training samples  | Tumour samples           |
| ttg-breast | Average of the normal training samples  | Tumour samples           |
| ttg-all    | Average of the normal training samples  | Tumour samples           |

## References
[1] The data come from the [TCGA Research Network](https://www.cancer.gov/tcga), the [TARGET Research Network](www.cancer.gov/ccg/research/genome-sequencing/target) and the [GTEx Research Network](https://gtexportal.org/home/). 

## Contact
Please contact us if there are any problems.

Myriam Bontonou (myriam.bontonou@ens-lyon.fr)
