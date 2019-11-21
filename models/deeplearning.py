import h5py
import torch
from torch.utils import data
from hdf5_dataset import HDF5Dataset

num_epochs = 50
loader_params = {'batch_size': 100, 'shuffle': True, 'num_workers': 6}

dataset = HDF5Dataset('./data/', recursive=True, load_data=False, 
   data_cache_size=4, transform=None)

data_loader = data.DataLoader(dataset, **loader_params)

"""for i in range(num_epochs):
   for x,y in data_loader:
      # here comes your training loop
      pass"""