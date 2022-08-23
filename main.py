# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn

# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

# from data import ClimateDataset
# from model import FCNetwork
# from utils import save_checkpoint
# from train import train, validate

# # 10. deep learning for timeseries
# train_loader = DataLoader(ClimateDataset('data_file/jena_climate/train.csv'),
#     batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
# val_loader = DataLoader(ClimateDataset('data_file/jena_climate/val.csv'),
#     batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

# best_loss = 1e+10
# directory = '../drive/MyDrive/pytorch-tutorial-log/'
# name = 'FCNetwork'
# writer = SummaryWriter(directory + name)

# start_epoch = 0
# epochs = 10

# model = FCNetwork(1680)
# model = model.cuda()

# criterion = nn.L1Loss().cuda() #MAE
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*epochs)

# for epoch in range(start_epoch, epochs):
#     train(train_loader, model, criterion, optimizer, scheduler, writer, epoch)
#     loss = validate(val_loader, model, criterion, writer, epoch)

#     is_best = loss < best_loss
#     best_prec1 = min(loss, best_loss)
#     save_checkpoint({
#         'epoch': epoch + 1,
#         'model': model.state_dict(),
#         'optimizer' : optimizer.state_dict(),
#         'scheduler' : scheduler.state_dict(),
#         'best_loss': best_loss,
#     }, is_best, directory + name + '/')
# print('Best loss: ', best_loss)


# 11. deep learning for text
import torch
from torch import nn
import time
from data import get_IMDB_data
from model import TextClassificationModel

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()
    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(torch.flatten(predicted_label), label.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = torch.flatten(model(text, offsets))
            loss = criterion(predicted_label, label.float())
            total_acc += (torch.abs(predicted_label-label)<0.5).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


# Hyperparameters
EPOCHS = 10 # epoch
LR = 5  # learning rate
BATCH_SIZE = 64 # batch size for training

vocab_size, train_dataloader, valid_dataloader, test_dataloader = get_IMDB_data(BATCH_SIZE)

emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).cuda()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None


for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)


#12. generative deep learning

