import time
import numpy as np

import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from data_gen import VaeDataset
from models import SegNet
from utils import *

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as T
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train-data-path', '-train', type=str, default='data/cifar-10/cifar10_train', required=True,
                    help='training data path: data/cifar-10/cifar10_train')
parser.add_argument('--valid-data-path', '-valid', type=str, default='data/cifar-10/cifar10_train', required=True,
                    help='valid data path: data/cifar-10/cifar10_train')
parser.add_argument('--batch_size', '-b', type=int, default=256, required=True, help='batch size. default=256')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001, required=True,
                    help='learning rate. default=0.0001')
parser.add_argument('--epochs', '-e', type=int, default=120, required=True, help='epochs. default=120')
parser.add_argument('--noise_level', '-n', type=float, default=0.15, required=True,
                    help='noise injection level. default=0.15')
parser.add_argument('--num_class', '-c', type=int, default=10, required=True,
                    help='the number of classes to be classified')

args = parser.parse_args()


def train(epoch, train_loader, model, optimizer, num_classes):
    # Ensure dropout layers are in train mode
    model.train()

    # Loss function
    # criterion = nn.MSELoss().to(device)

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    # losses = ExpoAverageMeter()  # loss (per word decoded)

    start = time.time()

    # to draw train loss graph
    weighted_loss_list = []

    # Batches
    for i_batch, (x, y) in enumerate(train_loader):
        # to update weighted loss, initialize
        loss_list = []
        # to print loss to terminal
        loss_print = []

        # Set device options
        x = x.to(device)
        y = y.to(device)
        y_onehot = encode_onehot(y, num_classes)

        # print(np.array(x.size()))
        # >>> [batch size, # of channels, img_width, img_height]

        # Zero gradients
        optimizer.zero_grad()

        # model output
        x_hat = model(x)

        # for every batch, calculate loss and update weighted loss
        for i in range(np.array(x.size())[0]):
            # RMSE (Root MSE) loss
            loss = torch.sqrt((x_hat[i, :, :, :] - x[i, :, :, :]).pow(2).mean())
            loss_list.append(loss)

        # sort: 내림차순
        loss_list.sort(reverse=True)

        # high loss has large weight
        for i in range(len(loss_list)):
            weighted_loss = loss_list[i] * (1/loss_list[-(i + 1)])
            weighted_loss = loss_list[i]
            weighted_loss_list.append(weighted_loss.item())
            loss_print.append(weighted_loss.item())
            # convert value to tensor and backprop
            # actually, it updates for every sample, not every batch.
            weighted_loss = torch.tensor(weighted_loss).requires_grad_(True)
            weighted_loss.backward()


        # loss.backward()

        # def closure():
        #     optimizer.zero_grad()
        #     y_hat = model(x)
        #     loss = torch.sqrt((y_hat - y).pow(2).mean())
        #     loss.backward()
        #     losses.update(loss.item())
        #     return loss

        # optimizer.step(closure)
        optimizer.step()

        # Keep track of metrics
        # losses.update(loss.item())
        batch_time.update(time.time() - start)

        start = time.time()

        loss_print_mean = sum(loss_print) / len(loss_print)

        # Print status
        if i_batch % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss}\t'.format(epoch, i_batch, len(train_loader),
                                         batch_time=batch_time, loss=loss_print_mean))

    # to draw loss graph, output mean of weighted loss
    total_loss = sum(weighted_loss_list) / len(weighted_loss_list)

    return total_loss


def valid(val_loader, model):
    model.eval()  # eval mode (no dropout or batchnorm)

    # Loss function
    # criterion = nn.MSELoss().to(device)

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)

    start = time.time()

    with torch.no_grad():
        # Batches
        for i_batch, (x, y) in enumerate(val_loader):
            # Set device options
            x = x.to(device)
            y = y.to(device)

            x_hat = model(x)

            loss = torch.sqrt((x_hat - x).pow(2).mean())

            # Keep track of metrics
            losses.update(loss.item())
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i_batch % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i_batch, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))

    return losses.avg


def main():
    train_data = ImageFolder(root=args.train_data_path, transform=T.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_data = ImageFolder(root=args.valid_data_path, transform=T.ToTensor())
    val_loader = DataLoader(dataset=valid_data, batch_size=len(valid_data), shuffle=False, pin_memory=True)

    # Create SegNet model
    model = SegNet(in_channels=3, is_unpooling=True, noise_level=args.noise_level, num_classes=args.num_class)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

    best_loss = 100000
    epochs_since_improvement = 0

    # draw loss graph
    epoch_graph = []
    train_loss_graph = []
    val_loss_graph = []
    auto_loss_graph = []
    class_loss_graph = []

    # Epochs
    for epoch in range(args.epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train_loss = train(epoch, train_loader, model, optimizer)

        # One epoch's validation
        val_loss = valid(val_loader, model)
        print('\n * LOSS - {loss:.3f}\n'.format(loss=val_loss))

        # Check if there was an improvement
        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # draw loss graph
        epoch_graph.append(epoch + 1)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, val_loss, is_best)


if __name__ == '__main__':
    main()
