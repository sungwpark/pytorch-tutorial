import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import ClimateDataset
from model import FCNetwork
from utils import save_checkpoint
from train import train, validate

# 10. deep learning for timeseries
train_loader = DataLoader(ClimateDataset('data_file/jena_climate/train.csv'),
    batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(ClimateDataset('data_file/jena_climate/val.csv'),
    batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

best_loss = 1e+10
directory = 'drive/MyDrive/pytorch-tutorial-log/'
name = 'FCNetwork'
writer = SummaryWriter(directory + name)

start_epoch = 0
epochs = 10

model = FCNetwork(1680)
model = model.cuda()

criterion = nn.L1Loss().cuda() #MAE
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*epochs)

for epoch in range(start_epoch, epochs):
    train(train_loader, model, criterion, optimizer, scheduler, writer, epoch)
    loss = validate(val_loader, model, criterion, writer, epoch)

    is_best = loss < best_loss
    best_prec1 = max(loss, best_loss)
    save_checkpoint({
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict(),
        'best_loss': best_loss,
    }, is_best, directory + name + '/')
print('Best loss: ', best_loss)

