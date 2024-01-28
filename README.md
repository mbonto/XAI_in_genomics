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
The datasets are stored in a folder on your computer. Set the absolute path of this folder in the function set_path in setting.py.

**PanCan** Go to the Pancan/Data folder and execute `python get_pancan.py`.

**BRCA** Go to the Gdc/Data folder and execute `python get_gdc.py`.

**BRCA-pam** Go to the Legacy/Data folder and execute `python get_legacy.py`.

**ttg-breast and ttg-all** Go to the TTG/Data folder and execute `python get_ttg.py`.

*The datasets gather data coming from the TCGA, TARGET and GTEx databases [1]. More details on the datasets can be found in the `Describe_data.ipynb` and `Discover_gene_expression_data.ipynb` notebooks in their respective folders.*

**Simulations** A code to simulate data from a latent dirichlet allocation model is also accessible in the Simulation/Data folder. 

### 3. Prepare datasets
**The same commands can be used for multiple dataset_names - pancan (PanCan), BRCA, BRCA-pam, ttg-breast, ttg-all.**

To access the data, a torch dataset is defined by the custom class [TCGA_dataset(data_path, database, cancer, label_name, weakly_expressed_genes_removed=True, ood_samples_removed=True, normalize_expression=True)](dataset.py).

Two functions use this class.
- [Dataloader for PyTorch](loader.py): train_loader, test_loader, n_class, n_feat, class_name, feat_name, transform, n_sample = load_dataloader(data_path, name, device, weakly_expressed_genes_removed=True, ood_samples_removed=True). *transform is a function standardising gene values using their means and standard deviations calculated from the training data.*
 
- [Dataset for scikit-learn](loader.py): X_train, X_test, y_train, y_test, n_class, n_feat, class_name, feat_name = load_dataset(data_path, name, normalize, weakly_expressed_genes_removed=True, ood_samples_removed=True, studied_features=None, normalize_expression=True). *Each gene is standardised using its mean and standard deviation computed from the training data.*

#### Gene expression unit
Initially, genes in different datasets are not expressed with the same unit. Here, they are all expressed in log2(norm_count + 1), where norm_count indicates that the sum of the expression of all the genes in a sample is equal to 10^6.

| Dataset          | Original unit                | Unit used here           |
|:----------------:|:----------------------------:|:------------------------:|
| ttg-breast/all   | log2(count_uq + 1)           | log2(norm_count + 1)     |
| BRCA             | log2(count + 1)              | log2(norm_count + 1)     |
| pancan           | count_uq                     | log2(norm_count + 1)     |
| BRCA-pam         | log2(count_uq + 1)           | log2(norm_count + 1)     |


#### Quality control
By default, genes whose values are missing or whose maximum expression value is zero are deleted.
Additionally, low expressed genes (less than 5 counts in more than 75% training samples for each class) and out_of-distribution samples (in which more than 75% of genes have a zero expression) can be removed. To detect these genes and samples, go to Script/Preprocessing and execute `python quality_control -n [dataset_name]`. 

To save gene names in a text file, execute `python store_gene_names -n [dataset_name]`.   


### 4. Learn models
**The same commands can be used for multiple machine learning model_names, using PyTorch - logistic regression (LR), multilayer perceptron (MLP), graph neural network (GCN) - or scikit-learn - logistic regression (LR_L1_penalty, LR_L2_penalty).**

Go to Scripts/Model.

#### Graph
*k is a parameter limiting the density of edges in the graph. Only the edges with the highest n_node x k weights are kept.*

To compute the correlation graph over all features using the training data, execute `python infer_graph.py -n [dataset_name] --method pearson_correlation -k [integer]`.

#### Model
*exp is the experiment number used to initialise the parameters of the models and to store the results.*

To train a torch model, execute `python train_nn.py -n [dataset_name] -m [model_name] --exp [integer]`.

To train a scikit-learn model, execute `python train_sklearn.py -n [dataset_name] -m [model_name] --exp [integer]`.

The performance of a trained model is averaged over several experiments, indexed between 1 and n_repet. It is accessible with the command `python get_summary.py -n [dataset_name] -m [model_name] --n_repet [integer]`.  

### 5. Explain models
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
Each model is trained 10 times with a different random initialisation. The results presented here are the average balanced accuracies (%) and standard deviations obtained with the 10 learned models.

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
