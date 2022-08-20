import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

# climate_data = pd.read_csv('data_file/jena_climate/train.csv')
# climate_data = climate_data.iloc[1:]

# print(climate_data.mean(), climate_data.std())

class ClimateDataset(Dataset):
    def __init__(self, csv_file, sequence_length=120, sampling_rate=6):
        self.csv_file = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.sampling_rate = sampling_rate
    
    def __len__(self):
        return len(self.csv_file) - (self.sequence_length+1)*self.sampling_rate + 1
    
    def __getitem__(self, index):
        mean = np.array([988.749258, 8.825983, 282.905155, 4.313381,
            75.872672, 13.145746, 9.194171, 3.951500, 5.810526, 9.302119,
            1218.451615, 2.149780, 3.560489, 176.440638])
        std = np.array([8.505132, 8.770948, 8.865565, 7.080088,
            16.628815, 7.601302, 4.146895, 4.769626, 2.632772, 4.199687,
            42.039024, 1.533593, 2.323064, 85.850713])

        X = np.zeros((self.sequence_length, 14))
        for i in range(self.sequence_length):
            X[i] = np.array(self.csv_file.iloc[index + i*self.sampling_rate, 1:])
        X -= mean
        X /= std

        y = self.csv_file.iloc[index + self.sequence_length*self.sampling_rate, 2]

        return torch.tensor(X).float(), torch.tensor([y]).float()
