import os
from six.moves import urllib
import pandas as pd
import magic
import matplotlib.pyplot as plt
import datetime


def create_new_folder(path):
    """Create a folder in 'path' if it does not exist yet."""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == 17:
            pass
        else:
            raise


def read_file(file_path):
    """Return a list where each line of the file appears as a string.""" 
    with open(file_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def set_pyplot():
    """Set the default plt setting."""
    FONT_SIZE = 16
    plt.rc('font', size=FONT_SIZE+8)            # fontsize of the text sizes
    plt.rc('axes', titlesize=FONT_SIZE+8)       # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE+8)       # fontsize of the x and y labels
    plt.rc('legend', fontsize=FONT_SIZE+8)      # fontsize of the legend
    
    
def write_info(file_path, url):
    with open(file_path, 'w') as f:
        f.write(f"Url: {url}\n")
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        f.write(f"Access time: {date}")
        
        
def download(url, file_path):
    """Download a url at a given file_path if the file does not exist yet."""
    if os.path.isfile(file_path):
        print(file_path + ' already existing.')
    else:
        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        
        # Save data.
        with open(file_path, 'wb') as f:
            f.write(data.read())
        
        # Convert the 'Excel'/'ASCII text' file into a gzip tsv file.
        data_type = magic.from_file(file_path)
        print(data_type)
        if 'Excel' in data_type:
            df = pd.read_excel(file_path, index_col='Unnamed: 0')
            df.to_csv(file_path, index=None, header=True, sep='\t', compression='gzip')

        if 'ASCII text' in data_type:
            df = pd.read_csv(file_path, compression=None, sep='\t')
            df.to_csv(file_path, index=None, header=True, sep='\t', compression='gzip')
        
        # Assert that the download has been successful.
        if os.stat(file_path).st_size == 0:
            os.remove(file_path)
            error = IOError('Downloading {} failed.'.format(url))
            raise error
        
        # Store the url and the access time.
        write_info(file_path.split('.')[0]+'_info.txt', url)
