# XAI_in_genomics

This repository contains the code of the experiments performed in the following paper\
[A Comparative Analysis of Gene Expression Profiling
by Statistical and Machine Learning Approaches]()
by Myriam Bontonou, Anaïs Haget, Maria Boulougouri, Benjamin Audit, Pierre Borgnat, Jean-Michel Arbona.


## Abstract
Many machine learning models have been proposed to classify phenotypes from gene expression data. In addition to their good performance, these models have the potential to provide some understanding of phenotype by extracting explanations for their decisions. These explanations often take the form of a list of genes ranked in order of importance for the predictions, the best-ranked genes being interpreted as linked to the phenotype. We discuss the biological and the methodological limitations of such explanations. Experiments are performed on several datasets gathering cancer and healthy tissue samples from the TCGA, GTEx and TARGET databases. A collection of machine learning models including logistic regression, multilayer perceptron, and graph neural network are trained to classify samples according to their cancer type. Gene rankings are obtained from explainability methods adapted to these models, and compared to the ones from classical statistical feature selection methods such as mutual information, DESeq2, and edgeR. Interestingly, on simple tasks, we observe that the information learned by black-box neural networks is related to the notion of differential expression. In all cases, a small set containing the best-ranked genes is sufficient to achieve a good classification. However, the top-ranked genes differ significantly between the methods and similar classification performance can be achieved with lower ranked genes. In conclusion, although these methods enable to identify biomarkers characteristic of certain pathologies, our results question the completeness of the selected gene sets and thus of explainability by the identification of the underlying biological processes.

## Usage
### 1. Dependencies
<details>
<summary> <b> Show </b> </summary><br>
 
- Python = 3.7
- PyTorch = 1.11
- PyTorch geometric = 2.0

</details>

### 2. Download datasets
<details>
<summary> <b> Show </b> </summary><br>
 
The datasets are stored in a folder on your computer. Set the absolute path of this folder in the function set_path in setting.py.

**PanCan** Go to the Pancan/Data folder and execute `python get_pancan.py`.

**BRCA** Go to the Gdc/Data folder and execute `python get_gdc.py`.

**BRCA-pam** Go to the Legacy/Data folder and execute `python get_legacy.py`.

**ttg-breast and ttg-all** Go to the TTG/Data folder and execute `python get_ttg.py`.

*The datasets gather data coming from the TCGA, TARGET and GTEx databases [1]. More details on the datasets can be found in the `Describe_data.ipynb` and `Discover_gene_expression_data.ipynb` notebooks in their respective folders.*

**Simulations** A code to simulate data from a latent dirichlet allocation model is also accessible in the Simulation/Data folder. 
</details>
 
### 3. Prepare datasets
<details>
<summary> <b> Show </b> </summary><br>
 
**The same commands can be used for multiple dataset_names - pancan (PanCan), BRCA, BRCA-pam, ttg-breast, ttg-all.**

To access the data, a torch dataset is defined by the custom class [TCGA_dataset(data_path, database, cancer, label_name, weakly_expressed_genes_removed=True, ood_samples_removed=True, normalize_expression=True)](dataset.py).

Two functions use this class.
- [Dataloader for PyTorch](loader.py): train_loader, test_loader, n_class, n_feat, class_name, feat_name, transform, n_sample = load_dataloader(data_path, name, device, weakly_expressed_genes_removed=True, ood_samples_removed=True). *transform is a function standardising gene values using their means and standard deviations calculated from the training data.*
 
- [Dataset for scikit-learn](loader.py): X_train, X_test, y_train, y_test, n_class, n_feat, class_name, feat_name = load_dataset(data_path, name, normalize, weakly_expressed_genes_removed=True, ood_samples_removed=True, studied_features=None, normalize_expression=True). *Each gene is standardised using its mean and standard deviation computed from the training data.*

#### Gene expression unit
Initially, genes in different datasets are not expressed with the same unit. Here, they are all expressed in $log_2(norm_{count} + 1)$, where $norm_{count}$ indicates that the sum of the expression of all the genes in a sample is equal to 10^6.

| Dataset          | Original unit                | Unit used here           |
|:----------------:|:----------------------------:|:------------------------:|
| ttg-breast/all   | $log_2(count_{uq} + 1)$        | $log_2(norm_{count} + 1)$   |
| BRCA             | $log_2(count + 1)$           | $log_2(norm\_{count} + 1)$   |
| pancan           | $count_{uq}$                   | $log_2(norm\_{count} + 1)$   |
| BRCA-pam         | $log_2(count_{uq} + 1)$        | $log_2(norm\_{count} + 1)$   |


#### Quality control
By default, genes whose values are missing or whose maximum expression value is zero are deleted.
Additionally, low expressed genes (less than 5 counts in more than 75% training samples for each class) and out_of-distribution samples (in which more than 75% of genes have a zero expression) can be removed. To detect these genes and samples, go to Script/Preprocessing and execute `python quality_control -n [dataset_name]`. 

To save gene names in a text file, execute `python store_gene_names -n [dataset_name]`.   

</details>

### 4. Learn models
<details>
<summary> <b> Show </b> </summary><br>

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

</details>

### 5. Explain models
<details>
<summary> <b> Show </b> </summary><br>

Go to Scripts/Explanation.

The explanation of a model's prediction on a training example is elucidated through the Integrated Gradients method (IG).
- For a PyTorch model: `python get_attributions.py -n [dataset_name] -m [model_name] --exp [integer] --set train`.
- For a scikit-learn model: `python get_attributions_sklearn.py -n [dataset_name] -m [model_name] --exp [integer] --set train`.
- Scores averaged over all studied classes: `python get_attributions_averaged_per_class.py -n [dataset_name] -m [model_name] --exp [integer] --set train`.

LR can also be interpreted by looking at the amplitude of the parameters. 

`python get_LR_weights.py -n [dataset_name] -m [model_name] --exp [integer]`

#### Understand IG scores
The prediction gaps (PGs) can be used to analyse the IG scores. Local PGs are obtained by ranking the features of each example independently. Global PGs are obtained by ranking them in the same way for all examples of the same class. 
- For the PyTorch model, execute `python get_prediction_gaps.py -n [dataset_name] -m [model_name] --set train`.
- For the scikit-learn model, execute `python get_prediction_gaps_sklearn.py -n [dataset_name] -m [model_name] --set train`.

To see the results, averaged over several experiments (indexed between 1 and n_repet), execute `python save_PG_to_csv.py -n [dataset_name] -m [model_name] --n_repet [integer]`. The results are accessible in dataset_folder/Results/figures. 

</details>

### 6. Select relevant genes with other feature selection methods
<details>
<summary> <b> Show </b> </summary><br>

Go to Scripts/Model.

To attribute a score to each gene with variance (VAR), PCA and mutual information (MI), execute 

`python select_features_with_various_methods.py -n [dataset_name]`. 

To run edgeR and DESeq2, execute `python select_features_with_r.py -n [dataset_name]`. 

*Warning: these methods are coded in R packages. The rpy2 Python module must be installed to run them in a Python script.*

</details>

### 7. Overview of the metrics employed for ranking genes according to their level of importance
<details>
<summary> <b> Show </b> </summary><br>
 
| Method | Scores used to rank gene in order of importance | Multi-class adjustment (if needed)|
|:------:|:------:|:---------------------------------:|
| ML model (IG) | Absolute integrated gradients scores | Class-wise average score |
| LR (weight) | Absolute parameter values | Class-wise average score |
| EdgeR | - $log_{10}$ (adjusted p-values) | Highest score among pair-wise comparisons |
| DESeq2 | - $log_{10}$ (adjusted p-values) | Highest score among pair-wise comparisons |
| VAR | Variances | |
| PCA | Absolute values of the coefficient on the first PC | |
| MI | Mutual information | |

For reproducibility, the scores computed and used in the article are accessible in dataset_folder/Results/scores.zip. 

Go to Scripts/Explanation.

Execute `python generate_file_for_GSEA.py -n [dataset_name] -m [model_name] --exp [integer]` to save all these scores in dataset_folder/Results/GSEA.

Go to Scripts/Model.

To compare the top 10 and top 100 ranked genes, execute `python plot_selected_features.py -n [dataset_name]`. The results are accessible in dataset_folder/Results/figures.

</details>

### 8. Evaluation of the classification potential of genes selected through various methods
<details>
<summary> <b> Show </b> </summary><br>

Go to Scripts/Model.

To train a torch model with a subset of genes (n_feat_selected) selected by a method, execute 

`python train_nn.py -n [dataset_name] -m [model_name] --exp [integer] --selection [method] --n_feat_selected [integer] --selection_type [best, worst]`

To train a scikit-learn model, execute 

`python train_sklearn.py -n [dataset_name] -m [model_name] --exp [integer] --selection [method] --n_feat_selected [integer] --selection_type [best, worst]`.

The name of the methods can be: var, PCA_PC1, MI, IG_LR_L1_penalty_set_train_exp_1, edgeR, DESeq2, IG_LR_set_train_exp_1, IG_MLP_set_train_exp_1, IG_GCN_set_train_exp_1...

After retraining LR_L1_penalty, LR_L2_penalty, MLP and GCN on genes selected by IG, the results can be summarised by executing `python get_summary_FS_self.py -n [dataset_name] -m [model_name] --n_repet [integer]`. After retraining a MLP on genes selected by LR_L1_penalty, LR_L2_penalty, MLP, GCN, var, edgeR, DESeq2, MI and PCA, the results can be summarised by executing `python get_summary_FS_other.py -n [dataset_name] -m [model_name] --n_repet [integer]`. The results are accessible in dataset_folder/Results/model_name.

</details>

### 9. Evaluation of the biological potential of genes selected through various methods
<details>
<summary> <b> Show </b> </summary><br>

Go to Visualisation/.

Established genes sets that are over-represented in the top-ranked genes selected by the different methods, can be identified using the [GSEA website](https://www.gsea-msigdb.org/gsea/msigdb). For the experiments in this article, we stored the over-represented genes in a csv file. For reproducibility, these files are accessible in GSEA.zip. The figures of the article can be reproduced using show_GSEA.ipynb notebook.

</details>

## Results
<details>
<summary> <b> Show </b> </summary><br>

For more details, please have a look at the scientific article.

### 1. Datasets
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



### 3. Explaining with IG
The scores attributed to the variables are computed with IG for each example correctly classified of the training set. The importance of the value of a variable for a prediction is computed with respect to a default prediction on a reference example (called baseline). 

|  Name      |            Baseline                     |     Studied classes      | 
|:----------:|:---------------------------------------:|:------------------------:|
| PanCan     | Average of the training samples         |       All                |
| BRCA       | Average of the normal training samples  | Tumour samples           |
| BRCA-pam   | Average of the normal training samples  | Tumour samples           |
| ttg-breast | Average of the normal training samples  | Tumour samples           |
| ttg-all    | Average of the normal training samples  | Tumour samples           |

Predictions gaps are shown in the article.

### 4. Comparison of the gene selected by statistical and machine learning
Heatmaps illustrating the percentage of common genes within the top 100 and top 10, plots showing classification performance after retraining models with a specific gene subset and plots showing the results of the over-representation analysis are included in the article.

The heatmaps can be plotted by running `python plot_selected_features.py -n [dataset_name]`. The over-representation analysis can be plotted using Visualisation/show_GSEA.ipynb notebook.

</details>

## References
[1] The data come from the [TCGA Research Network](https://www.cancer.gov/tcga), the [TARGET Research Network](www.cancer.gov/ccg/research/genome-sequencing/target) and the [GTEx Research Network](https://gtexportal.org/home/). 

## Contact
Please contact us if there are any problems.

Myriam Bontonou (myriam.bontonou@ens-lyon.fr)
