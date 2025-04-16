import os
import pandas as pd
import ntpath

def training_process():
    fetched_data = fetch_data()
    balanced_data = balance_data(fetched_data)

def balance_data(fetched_data):
    num_bins = 25
    samples_per_bin = 400
    hist, bins = np.histogram(data['steering'], num_bins)

def fetch_data():
    data_dir = '../camera'
    columns = ['forward', 'left', 'right']
    data = pd.read_csv(os.path.join(data_dir, 'training_data.csv'), names = columns)
    data['forward'] = data['forward'].apply(path_leaf)
    data['left'] = data['left'].apply(path_leaf)
    data['right'] = data['right'].apply(path_leaf)
    #for i in range(len(data)):
    #   print(i)

def path_leaf(path):
        head, tail = ntpath.split(path)
        return tail


training_process()
