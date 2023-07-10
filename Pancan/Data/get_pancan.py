# Libraries
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
from setting import *
from download_data import *


# Download
data_path = get_data_path('pancan')
database = 'pancan'
cancer = 'pancan'
download_dataset(data_path, database, cancer)
