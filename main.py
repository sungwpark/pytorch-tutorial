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
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchtext.datasets import IMDB
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from model import TextLSTMNet
from train import train_text, validate_text

best_acc = 0
directory = '../drive/MyDrive/IMDB_sentiment/'
name = 'LSTMNet'
writer = SummaryWriter(directory + name)

tokenizer = get_tokenizer('basic_english')

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
    return label_list.cuda(), text_list.cuda(), offsets.cuda()

vocab_size = len(vocab)
model = TextLSTMNet(vocab_size)
model = model.cuda()

start_epoch = 0
epochs = 10

criterion = nn.BCELoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_iter, test_iter = IMDB()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_loader = DataLoader(split_train_, batch_size=64,
                              shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(split_valid_, batch_size=64,
                              shuffle=True, collate_fn=collate_batch)

for epoch in range(start_epoch, epochs):
    train_text(train_loader, model, criterion, optimizer, scheduler, writer, epoch)
    prec1 = validate_text(val_loader, model, criterion, writer, epoch)

    is_best = prec1 > best_acc
    best_prec1 = max(prec1, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict(),
        'best_prec1': best_prec1,
    }, is_best, directory + name + '/')
print('Best accuracy: ', best_prec1)





#12. generative deep learning

