import torch
import time

from utils import AverageMeter, accuracy
from torch.nn.utils import clip_grad_norm_

def train(train_loader, model, criterion, optimizer, scheduler, writer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        output = model(input)
        loss = criterion(output, target)
        losses.update(loss.data.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses))

    writer.add_scalar('train_loss', losses.avg, epoch)

def validate(val_loader, model, criterion, writer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        with torch.no_grad():
            output = model(input)
        loss = criterion(output, target)
        losses.update(loss.data.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses))

    print(' * Loss {loss.avg:.3f}'.format(loss=losses))

    writer.add_scalar('val_loss', losses.avg, epoch)
    return losses.avg

def train_text(train_loader, model, criterion, optimizer, scheduler, writer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (label, text, offsets) in enumerate(train_loader):
        output = model(text, offsets)
        loss = criterion(output, label)

        prec1 = accuracy(output.data, label, topk=(1,))[0]
        losses.update(loss.data.item(), text.size(0))
        top1.update(prec1.item(), text.size(0))

        optimizer.zero_grad()
        loss.backward()
        #clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))
            
    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_acc', top1.avg, epoch)

def validate_text(val_loader, model, criterion, writer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (label, text, offsets) in enumerate(val_loader):
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, label)
        
        prec1 = accuracy(output.data, label, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    writer.add_scalar('val_loss', losses.avg, epoch)
    writer.add_scalar('val_acc', top1.avg, epoch)
    return top1.avg