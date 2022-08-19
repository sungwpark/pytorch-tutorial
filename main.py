import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import *
from model import *
from utils import *
from train import *

# 10. deep learning for timeseries

best_loss = 1e+10
name = 'FCNetwork'
writer = SummaryWriter("drive/MyDrive/pytorch-tutorial-log/%s"%(name))

start_epoch = 0
epochs = 50

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
    }, is_best, name, directory='/content/drive/MyDrive/pytorch-tutorial-log/')
print('Best loss: ', best_loss)

