# Libraries
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
from setting import *
from download_data import *


# Download
data_path = os.path.join(set_path(), "tcga")
database = "gdc"
## cancers = read_file(os.path.join(data_path, "cancers"))
cancers = ["BRCA", "KIRC"]
for cancer in cancers:
    download_dataset(data_path, database, cancer)