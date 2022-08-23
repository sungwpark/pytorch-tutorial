import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader


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



from torchtext.datasets import IMDB

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset


def get_IMDB_data(BATCH_SIZE):

    tokenizer = get_tokenizer('basic_english')
    train_iter = IMDB(split='train')

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])


    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: 0 if (x == 'neg') else 1

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)
    
    vocab_size = len(vocab)

    train_iter, test_iter = IMDB()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=True, collate_fn=collate_batch)
    
    return vocab_size, train_dataloader, valid_dataloader, test_dataloader



