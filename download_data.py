import os
from utils import create_new_folder, download



def get_url(data_path, database='gdc', cancer='pancan'):
    """Several databases are related to TCGA: TCGA Pan-Cancer (pancan), GDC TCGA, legacy TCGA, TCGA TARGET GTEx (TTG). Here are the locations of the different sources. TCGA Pan-Cancer and GDC are directly retrieved from the GDC website. Legacy TCGA and TTG are retrieved from the UCSC XENA browser.
    
    Parameters:
        data_path  --  Path towards a folder where the data will be stored.
        database  --  'pancan', 'ttg', 'gdc' or 'legacy'.
        cancer  --  If database is 'gdc' or 'legacy', cancer must be set to 'BRCA' or 'KIRC'. 
                    For other databases, all types of cancers are automatically selected.
                    For this pipeline, use cancer='pancan' when database='pancan' and cancer='all' when database='ttg'. These names will be used to load the dataset.
    """
    assert database in ['pancan', 'gdc', 'legacy', 'ttg'], "'database' must be either 'pancan', 'ttg', 'gdc' or 'legacy'."
    if database == 'pancan':
        assert cancer == 'pancan', "All types of cancers are automatically selected when database = 'pancan'. 'cancer' must be set to 'pancan'."
    elif database == 'ttg':
        assert cancer == 'all', "All types of cancers are automatically selected when database = 'ttg'. 'cancer' must be set to 'all'."
    url = {}
    url['pancan'] = {
        'phenotype': 'https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81',
        'expression_counts': 'http://api.gdc.cancer.gov/data/3586c0da-64d0-4b74-a449-5ff4d9136611',
    }

    url['ttg'] = {
        'phenotype': 'https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/TcgaTargetGTEX_phenotype.txt.gz',
        'expression_counts': 'https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/TcgaTargetGtex_RSEM_Hugo_norm_count.gz',
    }

    url['gdc'] = {
        'phenotype': 'https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-{}.GDC_phenotype.tsv.gz'.format(cancer),
        'expression_counts': 'https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-{}.htseq_counts.tsv.gz'.format(cancer),
        'ID_gene_mapping': 'https://gdc-hub.s3.us-east-1.amazonaws.com/download/gencode.v22.annotation.gene.probeMap',
    }

    url['legacy'] = {
        'phenotype': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.{}.sampleMap%2F{}_clinicalMatrix'.format(cancer, cancer),
        'expression_counts': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.{}.sampleMap%2FHiSeqV2.gz'.format(cancer),
        'ID_gene_mapping': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/probeMap%2Fhugo_gencode_good_hg19_V24lift37_probemap'
    }
    return url[database]



def download_dataset(data_path, database, cancer):
    # Create folders to store data if they do not exist yet.
    create_new_folder(os.path.join(data_path, database, 'phenotype'))
    create_new_folder(os.path.join(data_path, database, 'expression'))
    
    # Retrieve urls.
    url = get_url(data_path, database, cancer)
    
    # Download.
    download(url['phenotype'], os.path.join(data_path, database, 'phenotype', '{}_phenotype.tsv.gz'.format(cancer)))
    download(url['expression_counts'], os.path.join(data_path, database, 'expression', '{}_counts.tsv.gz'.format(cancer)))

