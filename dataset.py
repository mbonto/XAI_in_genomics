import os
import pandas as pd
import torch
import numpy as np


# Labels
def get_possible_classes(database, cancer):
    possible = {}
    possible['pancan-pancan'] = ["type",]
    possible['gdc-BRCA'] = ['pathologic_M', 'pathologic_N', 'pathologic_T', 'tumor_stage.diagnoses',
                            'age_at_initial_pathologic_diagnosis', 'sample_type.samples',
                            'primary_diagnosis.diagnoses', 'code.tissue_source_site', 'disease_type', 
                            'gender.demographic', 'morphology.diagnoses', 'vital_status.demographic']
    possible['gdc-KIRC'] = ['age_at_initial_pathologic_diagnosis', 'sample_type.samples',
                            'neoplasm_histologic_grade', 'primary_diagnosis.diagnoses']
    possible['gdc-LIHC'] = ['age_at_initial_pathologic_diagnosis', 'sample_type.samples',
                            'neoplasm_histologic_grade', 'primary_diagnosis.diagnoses']
    possible['gdc-OV'] = ['age_at_initial_pathologic_diagnosis', 'sample_type.samples',
                            'neoplasm_histologic_grade', 'primary_diagnosis.diagnoses']
    possible['gdc-LUAD'] = ['sample_type.samples',]
    possible['legacy-BRCA'] = ['PAM50Call_RNAseq', 'PAM50_mRNA_nature2012', 'sample_type']
    possible['ttg-ttg-all'] = ['_sample_type', '_primary_site']
    possible['ttg-ttg-breast'] = ['_sample_type', '_primary_site']
    try:
        possible_classes = possible[f"{database}-{cancer}"]
    except:
        pass
    return possible_classes


# Phenotype
def load_phenotype(data_path, database, cancer):
    column_name = get_column_name(data_path, database, cancer, 'phenotype')
    if database == "ttg":
        file_path = os.path.join(data_path, database, 'phenotype', '{}_phenotype.tsv.gz'.format("all"))
        df = pd.read_csv(file_path, compression="gzip", sep="\t", index_col=column_name, encoding="ISO-8859-1")
        df.replace('Primary Tumor', 'Tumor', inplace=True)
        df.replace('Primary Solid Tumor', 'Tumor', inplace=True)
        df.replace('Normal Tissue', 'Normal', inplace=True)
        df.replace('Solid Tissue Normal', 'Normal', inplace=True)
        if cancer == 'ttg-breast':
            df = df.loc[df['_primary_site'] == 'Breast']
    else:
        file_path = os.path.join(data_path, database, 'phenotype', '{}_phenotype.tsv.gz'.format(cancer))
        df = pd.read_csv(file_path, compression="gzip", sep="\t", index_col=column_name)
        if database == "legacy" and cancer == "BRCA":
            # Keep samples from primary tumors (tumor subtypes) and normal surrounding tissues (normal subtype)
            df_t = df.loc[df["sample_type"] == "Primary Tumor"]
            df_t = df_t.loc[df["PAM50Call_RNAseq"] != "Normal"]
            df_n = df.loc[df["sample_type"] == "Solid Tissue Normal"]
            df_n = df_n.loc[df["PAM50Call_RNAseq"] == "Normal"]
            df = pd.concat([df_t, df_n])
    return df


def get_unwanted_labels(database, cancer):
    unwanted = {}
    unwanted['pancan-pancan'] = []
    unwanted['ttg-ttg-all'] = ['Additional - New Primary', 'Additional Metastatic', 'Cell Line', 'Control Analyte', 'Metastatic', 'Post treatment Blood Cancer - Blood', 'Post treatment Blood Cancer - Bone Marrow', 'Primary Blood Derived Cancer - Bone Marrow', 'Primary Blood Derived Cancer - Peripheral Blood', 'Recurrent Blood Derived Cancer - Bone Marrow', 'Recurrent Blood Derived Cancer - Peripheral Blood', 'Recurrent Solid Tumor', 'Recurrent Tumor']
    unwanted['ttg-ttg-breast'] = ['Additional - New Primary', 'Additional Metastatic', 'Cell Line', 'Control Analyte', 'Metastatic', 'Post treatment Blood Cancer - Blood', 'Post treatment Blood Cancer - Bone Marrow', 'Primary Blood Derived Cancer - Bone Marrow', 'Primary Blood Derived Cancer - Peripheral Blood', 'Recurrent Blood Derived Cancer - Bone Marrow', 'Recurrent Blood Derived Cancer - Peripheral Blood', 'Recurrent Solid Tumor', 'Recurrent Tumor']
    unwanted['gdc-BRCA'] = ['not reported', 'nan', 'stage x', 'MX', 'NX', 'TX', 'Metastatic']
    unwanted['gdc-KIRC'] = ['Additional - New Primary']
    unwanted['gdc-LUAD'] = ['FFPE Scrolls', 'Recurrent Tumor']
    unwanted['legacy-BRCA'] = ['nan']
    try:
        unwanted_labels = unwanted[f"{database}-{cancer}"]
    except:
        unwanted_labels = []
    return unwanted_labels


def clean_labels(label_key, database, cancer):
    """Given a list of labels 'label_key', remove the unwanted ones."""
    unwanted_labels = get_unwanted_labels(database, cancer)
    for value in unwanted_labels:
        try :
            label_key.remove(value)
        except :
            pass
    # Remove nan numbers
    label_key = [label for label in label_key if label == label]
    return label_key


# Gene expression
def quality_control(data_path, database, cancer, df, weakly_expressed_genes_removed, ood_samples_removed):
    """
    Some samples and some genes are removed from the study.

    Here, we choose to remove:
      - genes whose expression value is missing for some samples,
      - genes whose maximal expression level is 0.

    Additionally,
      - samples listed in "ood_samples.npy" when ood_samples_removed is True,
      - genes listed in "low_expressed_genes_{cancer}.npy" when  weakly_expressed_genes_removed is True.
      # - genes listed in "constant_genes_{cancer}.npy".
    In this study, "low_expressed_genes_{cancer}.npy" lists the genes whose level is lower than 5 counts in more than 75% training samples of each class.
                   # "constant_genes_{cancer}.npy" lists the genes that are constant in the training dataset.
                   "ood_samples.npy" lists the samples whose more than 75% of genes have a zero expression. 
    To generate these files with different threshold, delete them and use the code in Script/Preprocessing/quality_control.py.
    """
    # Remove genes whose expression level is missing for some samples
    df.dropna(axis=1, inplace=True)
    # Remove genes whose maximal expression level is 0
    genes_to_remove = df.columns[df.max() <= 0]
    df.drop(columns=genes_to_remove, inplace=True)
    # Remove genes whose expression levels are low
    if weakly_expressed_genes_removed:
        try:
            genes_to_remove = list(np.load(os.path.join(data_path, database, "expression", f"low_expressed_genes_{cancer}.npy")))
            df.drop(columns=genes_to_remove, inplace=True)
            print(f"{len(genes_to_remove)} weakly expressed genes are removed of the dataset.")
        except:
            print(f"The file low_expressed_genes_{cancer}.npy has not been generated. Please, run quality_control.py stored in Scripts/Preprocessing.")
    # Remove genes whose expression levels are constant
    # if os.path.isfile(os.path.join(data_path, database, "expression", f"constant_genes_{cancer}.npy")):
    #     genes_to_remove = list(np.load(os.path.join(data_path, database, "expression", f"constant_genes_{cancer}.npy")))
    #     df.drop(columns=genes_to_remove, inplace=True)
    # Remove out-of-distribution samples
    if ood_samples_removed:
        try:
            samples_to_remove = list(np.load(os.path.join(data_path, database, "expression", f"ood_samples_{cancer}.npy")))
            if len(samples_to_remove) != 0:
                df.drop(index=samples_to_remove, inplace=True)
                print(f"{len(samples_to_remove)} samples are removed.")
        except:
            print(f"The file ood_samples_{cancer}.npy has not been generated. Please, run quality_control.py stored in Scripts/Preprocessing.")


def load_expression(data_path, database, cancer, weakly_expressed_genes_removed, ood_samples_removed):
    column_name = get_column_name(data_path, database, cancer, 'expression')
    if database == 'ttg':
        file_path = os.path.join(data_path, database, 'expression', '{}_counts.pkl'.format('all'))
        # Format of the original dataset changed from tsv.gz to pickle to speed up loading time
        if not os.path.isfile(file_path):  # format of the original dataset changed from tsv.gz to pickle to speed up loading time
            df = pd.read_csv(os.path.join(data_path, database, 'expression', '{}_counts.tsv.gz'.format('all')), compression="gzip", sep="\t", index_col=column_name)
            df.to_pickle(file_path)
        df = pd.read_pickle(file_path)
    else:
        file_path = os.path.join(data_path, database, 'expression', '{}_counts.tsv.gz'.format(cancer))
        df = pd.read_csv(file_path, compression="gzip", sep="\t", index_col=column_name)
    df = df.transpose()
    cols = [c for c in df.columns if c[:2] == '__']
    df = df.drop(labels=cols, axis=1)
    quality_control(data_path, database, cancer, df, weakly_expressed_genes_removed, ood_samples_removed)
    if database == 'pancan':
        # Keep samples from primary tumors only
        sample_IDs = df.index.values.tolist()
        df = df.loc[np.isin([s.split('-')[3][:2] for s in sample_IDs], ['01', '03'])]
    return df


# Useful
def remove_samples(sample_IDs, labels, label_key, database):
    """
    Remove samples from 'sample_IDs' if they have no label
    or if their label is not in label_key.
    """
    IDs_to_remove = []
    for ID in sample_IDs:
        try:
            if database == 'pancan':
                y = labels["-".join(ID.split("-")[:3])]
            else:
                y = labels[ID]
            if y not in label_key:
                IDs_to_remove.append(ID)
        except:
            IDs_to_remove.append(ID)
    for ID in IDs_to_remove:
        sample_IDs.remove(ID)
        

def get_column_name(data_path, database, cancer, data_type):
    column = {}
    column['pancan'] = {"expression": "gene_id", "phenotype": "bcr_patient_barcode"}
    column['ttg'] = {"expression": "sample", "phenotype": "sample"}
    column['gdc'] = {"expression": "Ensembl_ID", "phenotype": "submitter_id.samples", "methylation": "Composite Element REF"}
    column['legacy'] = {"expression": "sample", "phenotype": "sampleID"}
    try:
        return column[f"{database}"][data_type]
    except KeyError:
        raise KeyError("Please, have a look at the original files and add the name of the column corresponding to the genes IDs (expression), to the methylation sites IDs (methylation) or to the individuals IDs (phenotypes).")


# Loader
class TCGA_dataset(torch.utils.data.Dataset):
    "Create a dataset containing data from TCGA."
    def __init__(self, data_path, database, cancer, label_name, weakly_expressed_genes_removed=True, ood_samples_removed=True, normalize_expression=True):
        self.data_path = data_path
        self.database = database
        self.cancer = cancer
        self.label_name = label_name
        
        
        # Gene expression
        # Load
        self.expression = load_expression(data_path, database, cancer, weakly_expressed_genes_removed, ood_samples_removed)
        
        # Extract the IDs of the samples (corresponding to individuals)
        self.sample_IDs = self.expression.index.values.tolist()
        
        
        # Phenotype
        # Load
        phenotype = load_phenotype(data_path, database, cancer)
        
        # Retrieve the column corresponding to the labels to classify
        self.labels = phenotype[label_name]  
        
        # Remove unwanted labels and associate each remaining label with a number from 0 to number of classes - 1
        label_key = sorted(np.unique(list(self.labels.values)))
        self.label_key = clean_labels(label_key, database, cancer)
        if database == 'gdc' and cancer == 'BRCA':
            self.label_map = {'Solid Tissue Normal': 0, 'Primary Tumor': 1}
        else:
            self.label_map = dict(zip(self.label_key, range(len(self.label_key))))
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        
        # Remove the IDs of the samples which have no label
        remove_samples(self.sample_IDs, self.labels, self.label_key, database)

        # Remove the samples which have no label
        self.expression = self.expression[self.expression.index.isin(self.sample_IDs)]

        
        # Remove some of the genes (optional)
        # remove_genes(data_path, database, cancer, self.expression)

        # Extract the IDs of the genes
        self.genes_IDs = self.expression.columns.values.tolist()

        # Normalise the gene expression of each sample
        if normalize_expression:
            if database in ['gdc', 'ttg', 'legacy']:  # reverse log2 if needed
                self.expression = 2**self.expression - 1
            self.expression = self.expression.div(self.expression.sum(axis=1), axis=0)  # gene expression divided by the total expression in the sample
            self.expression = np.log2(self.expression * 10**6 + 1)


    def __len__(self):
        "Total number of samples in the dataset."
        return len(self.sample_IDs)

    def __getitem__(self, index):
        "Generate a sample from the dataset."
        # Select the sample ID.
        ID = self.sample_IDs[index]

        # Load the features of the sample and its label.
        X = self.expression.loc[ID].values
        X = torch.from_numpy(X).type(torch.float)
        if self.database == 'pancan':
            y = self.labels["-".join(ID.split("-")[:3])]
        else:
            y = self.labels[ID]
        y = self.label_map[y]
        y = torch.tensor(y)

        return X, y    



class custom_dataset(torch.utils.data.Dataset):
    """Create a dataset from data stored in a matrix X of shape [n_sample, n_feat] 
    and a class vector y of shape [n_sample].
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        "Total number of samples in the dataset."
        return len(self.X)

    def __getitem__(self, index):
        "Generate a sample from the dataset."
        return self.X[index], self.y[index]

