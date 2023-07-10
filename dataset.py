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
    try:
        possible_classes = possible[f"{database}-{cancer}"]
    except:
        pass
    return possible_classes


# Phenotype
def load_phenotype(data_path, database, cancer):
    file_path = os.path.join(data_path, database, 'phenotype', '{}_phenotype.tsv.gz'.format(cancer))
    column_name = get_column_name(data_path, database, cancer, 'phenotype')
    df = pd.read_csv(file_path, compression="gzip", sep="\t", index_col=column_name)
    return df


def get_unwanted_labels(database, cancer):
    unwanted = {}
    unwanted['pancan-pancan'] = []
    unwanted['gdc-BRCA'] = ['not reported', 'nan', 'stage x', 'MX', 'NX', 'TX', 'Metastatic']
    unwanted['gdc-KIRC'] = ['Additional - New Primary']
    unwanted['gdc-LUAD'] = ['FFPE Scrolls', 'Recurrent Tumor']
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
def load_expression(data_path, database, cancer):
    file_path = os.path.join(data_path, database, 'expression', '{}_counts.tsv.gz'.format(cancer))
    column_name = get_column_name(data_path, database, cancer, 'expression')
    df = pd.read_csv(file_path, compression="gzip", sep="\t", index_col=column_name)
    df = df.transpose()
    cols = [c for c in df.columns if c[:2] == '__']
    df = df.drop(labels=cols, axis=1)
    df = df[df.columns[df.max() > 0]]
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
    column['gdc'] = {"expression": "Ensembl_ID", "phenotype": "submitter_id.samples", "methylation": "Composite Element REF"}
    try:
        return column[f"{database}"][data_type]
    except KeyError:
        raise KeyError("Please, have a look at the original files and add the name of the column corresponding to the genes IDs (expression), to the methylation sites IDs (methylation) or to the individuals IDs (phenotypes).")


# Loader
class TCGA_dataset(torch.utils.data.Dataset):
    "Create a dataset containing data from TCGA."
    def __init__(self, data_path, database, cancer, label_name):
        self.data_path = data_path
        self.database = database
        self.cancer = cancer
        self.label_name = label_name
        
        
        # Gene expression
        # Load
        self.expression = load_expression(data_path, database, cancer)
        
        # Remove genes whose expression value is missing for some samples
        self.expression = self.expression.dropna(axis=1)
        
        # Extract the IDs of the samples (corresponding to individuals)
        self.sample_IDs = self.expression.index.values.tolist()
        
        # Extract the IDs of the genes
        self.genes_IDs = self.expression.columns.values.tolist()
        
        
        # Phenotype
        # Load
        phenotype = load_phenotype(data_path, database, cancer)
        
        # Retrieve the column corresponding to the labels to classify.
        self.labels = phenotype[label_name]  
        
        # Remove unwanted labels and associate each remaining label with a number from 0 to number of classes - 1.
        label_key = sorted(np.unique(list(self.labels.values)))
        self.label_key = clean_labels(label_key, database, cancer)
        self.label_map = dict(zip(self.label_key, range(len(self.label_key))))
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        
        
        # Remove the IDs of the samples which have no label.
        remove_samples(self.sample_IDs, self.labels, self.label_key, database)

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

