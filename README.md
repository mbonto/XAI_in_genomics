# XAI_for_genomics

This repository contains the code of the experiments performed in the following paper\
[Studying Limits of Explainability by Integrated Gradients for Gene Expression Models](https://arxiv.org/pdf/2303.11336v1.pdf)
by Myriam Bontonou, Anaïs Haget, Maria Boulougouri, Jean-Michel Arbona, Benjamin Audit, Pierre Borgnat.

This repository also contains the code of the additional results presented in the French article\
[Expliquer la classification d'expression de gènes par la méthode des gradients intégrés]().


## Abstract
Understanding the molecular processes that drive cellular life is a fundamental question in biological research. Ambitious programs have gathered a number of molecular datasets on large populations. To decipher the complex cellular interactions, recent work has turned to supervised machine learning methods. The scientific questions are formulated as classical learning problems on tabular data or on graphs, e.g. phenotype prediction from gene expression data. In these works, the input features on which the individual predictions are predominantly based are often interpreted as indicative of the cause of the phenotype, such as cancer identification.
Here, we propose to explore the relevance of the biomarkers identified by Integrated Gradients, an explainability method for feature attribution in machine learning. Through a motivating example on The Cancer Genome Atlas, we show that ranking features by importance is not enough to robustly identify biomarkers. As it is difficult to evaluate whether biomarkers reflect relevant causes without known ground truth, we simulate gene expression data by proposing a hierarchical model based on Latent Dirichlet Allocation models. We also highlight good practices for evaluating explanations for genomics data and propose a direction to derive more insights from these explanations.

## Usage
### 1. Dependencies
- Python = 3.7
- PyTorch = 1.11
- PyTorch geometric = 2.0


### 2. Datasets
The datasets will be stored in a folder on your computer. Set the absolute path of this folder in the function set_path in setting.py.

#### TCGA data
##### PanCan
To download the PanCan TCGA dataset [1], go to the Pancan/Data folder and execute `python get_pancan.py`.
More details on the data are presented in two notebooks `Describe_tcga_data.ipynb` and `Discover_gene_expression_data.ipynb`.

##### BRCA and KIRC
To download the BRCA and KIRC datasets, go to the Gdc/Data folder and execute `python get_gdc.py`.
More details on the data are presented in two notebooks `Describe_gdc_tcga_data.ipynb` and `Discover_gdc_tcga_gene_expression_data.ipynb`.

#### Simulation
To generate SIMU1/SIMU2 data, go to Simulation/Data folder and execute `python get_simu.py --name SIMU1 --size 9900` and `python get_simu.py --name SIMU2 --size 9900`.

To generate SimuA/SimuB/SimuC data, execute `python get_simu.py --name SimuA --size 1200`, `python get_simu.py --name SimuB --size 1200` and `python get_simu.py --name SimuC --size 1200`.


### 3. Learning models
In the following, the same commands can be used for various datasets and learning models.
- Various datasets can be used: PanCan TCGA (pancan), BRCA, KIRC, SIMU1/SIMU2, SimuA/SimuB/SimuC. 
- Various models can be trained: logistic regression (LR), multilayer perceptron (MLP), diffusion + logistic regression (DiffuseLR), diffusion + multilayer perceptron (DiffuseMLP).

Go to Scripts/Model.
#### Graph
To compute the correlation graph over all features using training examples, execute `python infer_graph.py -n pancan --min_value 0.5`.
All correlations whose absolute value is lower than `min_value` are set to 0.

#### Model
To train a logistic regression (LR) on TCGA data (pancan), execute `python train_nn.py -n pancan -m LR`.


### 4. Explainability
Go to Scripts/Explanation.

To compute the integrated gradients scores, execute `python get_attributions.py -n pancan -m LR --set train` for the training examples and `python get_attributions.py -n pancan -m LR --set test` for the test examples.

To compute the prediction gaps (PGs), execute `python get_prediction_gaps.py -n pancan -m LR --set test`. Local PGs are obtained by ranking the features of each example independently. Global PGs are obtained by ranking them in the same way for all examples of the same class.

To compute the curves, execute `python get_attributions_averaged_per_class.py -n pancan -m LR --set test` followed by `python get_curves.py -n pancan -m LR --set test --simu 100`.

To compute the feature agreement metrics on simulated data, execute `python get_ranking_metrics.py -n SIMU1 -m LR --set test`. To compute the diffused feature agreement metrics on simulated data, execute `python get_ranking_metrics.py -n SIMU1 -m LR --set test --diffusion`.

## Results
### 1. Datasets
For more details, please have a look at the scientific articles.

|  Name  | # classes | # samples (max / min per class)  | # variables |
|:---------:|:-----------:|:------------------------------------------:|:--------------:|
| pancan |      33      |               9680 (1095 / 36)               |     16335     |
|   BRCA  |       2       |               1210 (1097 / 113)             |     58274     |
|   KIRC   |       2       |                 606 (534 / 72)                  |     58233     |
| SIMU1 |      33      |                    9900 (300)                    |     15000     |
| SIMU2 |      33      |                     9900 (300)                   |     15000     |
| SimuA |       2       |                     1200 (600)                    |     50000     |
| SimuB |       2       |                 1200 (900 / 300)              |     50000     |
| SimuC |       2       |               1200 (1000 / 200)              |     50000     |

Here is some additional information about the simulated datasets.

|   Name  | # classes | # variables | # groups | Overlapping<br>groups|     # variable<br>per group    | # over-expressed<br>groups per class | # subclass | # informative<br>variables |
|:-----:|:-----------:|:-----------:|:--------:|:--------------------:|:------------------------------:|:------------------------------------:|:----------:|:--------------------------:|
| SIMU1 |      33     |    15000   |    1500    |        No           |           10                   |               37 per class           |       -           |        370          |
| SIMU2 |      33     |    15000   |    3000    |       Yes           |      10 in average             |               37 per class           |        -          |        370          |
| SimuA |       2     |    50000   |    5000    |         No          |          10                    |              500 per class           |       1 / 1       |   10000 (class 0)   |
| SimuB |       2     |    50000   |    5000    |         No          |          10                    |          500 per subclass            |        3 / 1      |   10000 (class 0)   |
| SimuC |       2     |    50000   |    5000    |         No          |          10                    |          500 per subclass            |        5 / 1      |   10000 (class 0)   |

### 2. Learning
Each model is trained 10 times with a different random initialisation. The results presented here are the averages and standard deviations obtained with the 10 learned models.

**Logistic regression (LR)**

|  Name  | Training accuracy (%) | Balanced test accuracy (%)  | 
|:---------:|:---------------------------:|:------------------------------------:|
| pancan |          100 +- 0.0          |                93.7 +- 0.4                |
|   BRCA  |          100 +- 0.0          |                96.6 +- 0.3                |
|   KIRC   |          100 +- 0.0          |                98.7 +- 0.2                |
| SIMU1 |          100 +- 0.0          |                99.8 +- 0.1                |
| SIMU2 |          100 +- 0.0          |                99.6 +- 0.1                |
| SimuA  |          100 +- 0.0          |                100 +- 0.0                |
| SimuB  |          100 +- 0.0          |                100 +- 0.0                |
| SimuC  |          100 +- 0.0          |                100 +- 0.0                |

**Multilayer perceptron (MLP)**

|  Name  | Training accuracy (%) | Balanced test accuracy (%)  | 
|:---------:|:---------------------------:|:------------------------------------:|
| pancan |          100 +- 0.0          |                94.5 +- 0.3                |
|   BRCA  |          100 +- 0.0          |                99.6 +- 0.0                |
|   KIRC   |          99.7 +- 0.2          |                99.6 +- 0. 7               |
| SIMU1 |          100 +- 0.0          |                99.9 +- 0.0                |
| SIMU2 |          100 +- 0.0          |                99.6 +- 0.1                |
| SimuA  |          100 +- 0.0          |                100 +- 0.0                |
| SimuB  |          100 +- 0.0          |                100 +- 0.0                |
| SimuC  |          100 +- 0.0          |                100 +- 0.0                |

### 3. Explaining
The scores attributed to the variables are computed with integrated gradients for each example correctly classified of the test set. The importance of the value of a variable for a prediction is computed with respect to a default prediction on a reference example (called baseline). 

|  Name  |            Baseline            |     Studied classes      | 
|:---------:|:---------------------------:|:--------------------------:|
| pancan |          Null vector          |                All                |
|   BRCA  | Average of the normal samples used for training | Tumour samples |
|   KIRC   | Average of the normal samples used for training | Tumour samples |
| SIMU1 |          Null vector          |                All                |
| SIMU2 |          Null vector          |                All                |
| SimuA  | Average of the class 1 samples used for training | Class 0 samples |
| SimuB  | Average of the class 1 samples used for training | Class 0 samples |
| SimuC  | Average of the class 1 samples used for training | Class 0 samples |

#### Prediction gaps
The prediction gaps (PG) are calculated by modifying the values of the variables in each example in a certain order. Either in descending order of importance (PGI), in ascending order of importance (PGU) or in random order (PGR). The PGRs are averaged over 30 trials. The values of the modified variables are replaced by those of the baseline variables. The results presented here are the averages obtained for all the examples studied.

**Logistic regression (LR)** 

|  Name        |  PGU (%) (&darr;)<br>Local |  PGU (%) (&darr;)<br>Global | PGI (%) (&uarr;)<br>Local | PGI (%) (&uarr;)<br>Global |PGR (%) (&darr;)  | 
|:------------:|:--------------------------:|:---------------------------:|:------------------------:|:--------------------------:|:------------------:|
| pancan       | 0.73 +- 0.71 | 9.39 +- 0.36 | 96.03 +- 0.15 | 33.39 +- 1.15 | 3.11 +- 0.24 |
|   BRCA       | 1.06 +- 0.02 | 1.85 +- 0.05 | 99.94 +- 0.03 | 99.92 +- 0.04 | 88.76 +- 0.23 |
|   KIRC       | 1.30 +- 0.09 | 2.74 +- 0.39 | 99.94 +- 0.02 | 99.93 +- 0.02 | 87.68 +- 0.27 |
| SIMU1        | 0.07 +- 0.0  | 4.58 +- 0.26 | 99.05 +- 0.01 | 42.4 +- 0.46 | 3.68 +- 0.1 |
| SIMU2        | 0.07 +- 0.0  | 1.86 +- 0.14 | 99.46 +- 0.02 | 71.07 +- 0.68 | 8.08 +- 0.46 |
| SimuA        | 5.3 +- 0.02  | 8.02 +- 0.04 | 95.06 +- 0.02 | 92.45 +- 0.04 | 51.22 +- 0.07 |
| SimuB        | 6.97 +- 0.15 | 10.98 +- 0.26 | 98.43 +- 0.04 | 97.68 +- 0.07 | 76.0 +- 0.34 |
| SimuC        | 6.85 +- 0.23 | 10.94 +- 0.21 | 99.24 +- 0.04 | 98.87 +- 0.05 | 84.52 +- 0.31 |

**Multilayer perceptron (MLP)**

|  Name        |  PGU (%) (&darr;)<br>Local |  PGU (%) (&darr;)<br>Global | PGI (%) (&uarr;)<br>Local | PGI (%) (&uarr;)<br>Global |PGR (%) (&darr;)  | 
|:------------:|:--------------------------:|:---------------------------:|:------------------------:|:--------------------------:|:------------------:|
| pancan       | 4.76 +- 1.73 | 17.17 +- 1.26 | 96.1 +- 0.24 | 59.04 +- 1.81 | 23.17 +- 0.57 |
|   BRCA       | 0.94 +- 0.2 | 1.56 +- 0.33 | 98.71 +- 0.27 | 98.22 +- 0.28 | 53.52 +- 2.06 |
|   KIRC       | 1.03 +- 0.2 | 1.32 +- 0.31 | 98.44 +- 0.46 | 98.04 +- 0.47 | 54.08 +- 2.46 |
| SIMU1        | 0.27 +- 0.0 | 20.01 +- 0.23 | 98.45 +- 0.06 | 49.91 +- 0.33 | 25.46 +- 0.19 |
| SIMU2        | 0.17 +- 0.0 | 8.6 +- 0.06 | 98.03 +- 0.08 | 73.1 +- 0.11 | 28.2 +- 0.13 |
| SimuA        | 5.47 +- 0.24 | 8.32 +- 0.32 | 94.87 +- 0.23 | 92.13 +- 0.3 | 51.14 +- 1.64 |
| SimuB        | 6.67 +- 0.27 | 9.32 +- 0.58 | 97.44 +- 0.18 | 96.4 +- 0.24 | 67.23 +- 1.8 |
| SimuC        | 5.72 +- 0.16 | 7.69 +- 0.2 | 97.38 +- 0.17 | 96.34 +- 0.25 | 64.36 +- 1.54 |

#### Feature agreements
The feature agreement (fA) measures the percentage of "important" variables identified by the integrated gradients method. For two-class classification problems, we assume that the important variables are those that allow to differenciate a class from the other. For problems with a larger number of classes, we assume that the variables that allow one class to be identified do not allow another class to be identified among the other classes.

For classes made up of distinct sub-classes, the feature agreement is calculated on the basis of important variables for each sub-class. 

**Logistic regression (LR)**

|  Name  | FA (%) (&uarr;)<br>Without diffusion<br>Local | FA (%) (&uarr;)<br>Without diffusion<br>Global | FA (%) (&uarr;)<br>With diffusion<br>Local | FA (%) (&uarr;)<br>With diffusion<br>Global | 
|:------:|:---------------------------------------------:|:-----------------------------------------------:|:------------------------------------------:|:--------------------------------------------:|
| SIMU1 | 72.84 +- 0.41 | 99.46 +- 0.08 | 77.67 +- 0.32 | 99.87 +- 0.08 |
| SIMU2 | 46.52 +- 0.86 | 88.13 +- 0.49 | 45.7 +- 0.56 | 69.82 +- 0.15 |
| SimuA  | 78.03 +- 0.35 | 99.86 +- 0.03 | 82.84 +- 0.29 | 100.0 +- 0.0 |
| SimuB  | 62.82 +- 0.87 | 87.75 +- 1.08 | 67.01 +- 0.88 | 90.78 +- 1.1 |
| SimuC  | 58.65 +- 1.01 | 79.49 +- 0.88 | 62.37 +- 0.82 | 82.16 +- 0.67 |

**Multilayer perceptron (MLP)**

|  Name  | FA (%) (&uarr;)<br>Without diffusion<br>Local | FA (%) (&uarr;)<br>Without diffusion<br>Global | FA (%) (&uarr;)<br>With diffusion<br>Local | FA (%) (&uarr;)<br>With diffusion<br>Global | 
|:------:|:---------------------------------------------:|:-----------------------------------------------:|:------------------------------------------:|:--------------------------------------------:|
| SIMU1 | 74.18 +- 0.26 | 100.0 +- 0.01 | 75.9 +- 0.26 | 100.0 +- 0.0 |
| SIMU2 | 45.95 +- 0.17 | 87.77 +- 0.22 | 44.75 +- 0.13 | 69.9 +- 0.13 |
| SimuA  | 79.94 +- 0.26 | 99.96 +- 0.02 | 83.28 +- 0.32 | 100.0 +- 0.0 |
| SimuB  | 69.14 +- 0.62 | 94.62 +- 0.48 | 72.32 +- 0.65 | 95.93 +- 0.4 |
| SimuC  | 67.81 +- 0.66 | 90.47 +- 0.65 | 71.04 +- 0.73 | 92.18 +- 0.74 |


## References
[1] The data come from the [TCGA Research Network](https://www.cancer.gov/tcga).

## Contact
Please contact us if there are any problems.

Myriam Bontonou (myriam.bontonou@ens-lyon.fr)
