# Libraries
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
from setting import *
from download_data import *


# Download
data_path = os.path.join(set_path(), "tcga")
database = "ttg"
cancers = ["all", ]
for cancer in cancers:
    download_dataset(data_path, database, cancer)
