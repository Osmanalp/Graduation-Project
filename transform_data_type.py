#Veri dönüşümü .npy-.csv
import pandas as pd
import numpy as np
#temiz EEG verileri için
data_array = np.load('data/cdata/EEG_all_epochs.npy')
data = pd.DataFrame(data_array)
data.to_csv('data/cdata/EEG_all_epochs.csv', index = None)
#EOG verileri için
data_array = np.load('data/ndata/EOG_all_epochs.npy')
data = pd.DataFrame(data_array)
data.to_csv('data/ndata/EOG_all_epochs.csv', index = None)
#EMG Verileri için
data_array = np.load('data/ndata/EMG_all_epochs.npy')
data = pd.DataFrame(data_array)
data.to_csv('data/ndata/EMG_all_epochs.csv', index = None)
# Gürülütülü EEG verileri için
data_array = np.load('data/ndata/EEG_all_epochs.npy')
data = pd.DataFrame(data_array)
data.to_csv('data/ndata/EEG_all_epochs.csv', index = None)