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
import matplotlib.pyplot as plt
import torch.nn.functional as F

import visdom
from io import BytesIO
from PIL import Image
import seaborn as sns # import this after torch or it will break everything


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
parser.add_argument('--alpha', '-a', type = float, default=0.5, required=True,
                    help='the rate of autoencoder loss and classifier loss')

args = parser.parse_args()

# for classifier
prediction_criterion = nn.NLLLoss().to(device)
vis = visdom.Visdom()
vis.env = 'confidence_estimation'
conf_histogram = None

def plot_histograms(corr, conf, bins=50, norm_hist=True):
    # Plot histogram of correctly classified and misclassified examples in visdom
    global conf_histogram

    plt.figure(figsize=(6, 4))
    sns.distplot(conf[corr], kde=False, bins=bins, norm_hist=norm_hist, label='Correct')
    sns.distplot(conf[np.invert(corr)], kde=False, bins=bins, norm_hist=norm_hist, label='Incorrect')
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.legend()

    # the image buffer acts as if it where a location on disk
    img_buffer = BytesIO()
    plt.savefig(img_buffer, bbox_inches='tight', pad_inches=0)
    img = Image.open(img_buffer)
    img = img.convert('RGB')
    img = torch.FloatTensor(np.array(img)).permute(2, 0, 1)
    conf_histogram = vis.image(img, win=conf_histogram, opts=dict(title='Confidence Histogram'))


def train(epoch, train_loader, model, optimizer, num_classes, correct_count, total):
    # Ensure dropout layers are in train mode
    model.train()

    # Loss function
    # criterion = nn.MSELoss().to(device)

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    # losses = ExpoAverageMeter()  # loss (per word decoded)

    start = time.time()

    # to draw train loss graph
    weighted_loss_list = []

    # for classifier
    lmbda = 0.1

    # Batches
    for i_batch, (x, y) in enumerate(train_loader):
        # to update weighted loss, initialize
        loss_list = []
        max_loss = 0.
        # to print loss to terminal
        loss_print = []

        # Set device options
        x = x.to(device)
        y = y.to(device)
        y_onehot = encode_onehot(y, args.num_class)

        # print(np.array(x.size()))
        # >>> [batch size, # of channels, img_width, img_height]

        # Zero gradients
        optimizer.zero_grad()

        # model output
        x_hat, classify, confidence = model(x)

        #############
        # Classifier
        #############
        classify = F.softmax(classify, dim=-1)
        confidence = torch.sigmoid(confidence)

        # Randomly set half of the confidences to 1 (i.e. no hints)
        b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).to(device)
        conf = confidence * b + (1 - b)
        classify_new = classify * conf.expand_as(classify) + y_onehot * (1 - conf.expand_as(y_onehot))
        classify_new = torch.log(classify_new)

        # classifier loss
        xentropy_loss = prediction_criterion(classify_new, y)
        confidence_loss = torch.mean(-torch.log(confidence))

        total_loss = xentropy_loss + (lmbda * confidence_loss)

        # 0.3 = hint budget
        if 0.3 > confidence_loss.data:
            lmbda = lmbda / 1.01
        else:
            lmbda = lmbda / 0.99

        # loss = torch.sqrt((x_hat - x).pow(2).mean())
        # loss.backward()
        # for every batch, calculate loss and update weighted loss
        for i in range(np.array(x.size())[0]):
            # RMSE (Root MSE) loss
            loss = (x_hat[i, :, :, :] - x[i, :, :, :]).pow(2).mean()
            max_loss = max(loss, max_loss)

        # # sort: 내림차순
        # loss_list.sort(reverse=True)
        loss_record = max_loss + args.alpha * total_loss
        loss = max_loss + args.alpha * total_loss
        loss.backward()

        # classifier
        pred_idx = torch.max(classify.data, 1)[1]
        total += y.size(0)
        correct_count += (pred_idx == y.data).sum()
        accuracy = correct_count / total

        # # high loss has large weight
        # for i in range(len(loss_list)):
        #     weighted_loss = loss_list[i] * (1 - 0.5 / len(loss_list) * i)
        #     weighted_loss_list.append(weighted_loss.item())
        #     loss_print.append(weighted_loss.item())
        #     # convert value to tensor and backprop
        #     # actually, it updates for every sample, not every batch.
        #     weighted_loss = torch.tensor(weighted_loss).requires_grad_(True)
        #     weighted_loss.backward()


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

        # loss_print_mean = sum(loss_print) / len(loss_print)

        # Print status
        if i_batch % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss:.5f}\t'
                  'Class accuracy {accuracy:.3f}\t'
                  .format(epoch, i_batch, len(train_loader), batch_time=batch_time,
                          loss=loss_record, accuracy=accuracy*100))

    # # to draw loss graph, output mean of weighted loss
    # total_loss = sum(weighted_loss_list) / len(weighted_loss_list)

    return loss_record, lmbda


def valid(val_loader, model, lmbda):
    model.eval()  # eval mode (no dropout or batchnorm)

    # Loss function
    # criterion = nn.MSELoss().to(device)

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)

    start = time.time()

    # for classifier
    correct = []
    confidence = []

    with torch.no_grad():
        # Batches
        for i_batch, (x, y) in enumerate(val_loader):
            # Set device options
            x = x.to(device)
            y = y.to(device)
            y_onehot = encode_onehot(y, args.num_class)

            x_hat, pred, conf_output = model(x)

            #############
            # classifier
            #############
            pred_val = F.softmax(pred, dim=-1)
            conf = torch.sigmoid(conf_output).data.view(-1)
            conf_val = torch.sigmoid(conf_output)

            pred_value, pred = torch.max(pred_val.data, 1)
            correct.extend((pred == y).cpu().numpy())
            confidence.extend(conf.cpu().numpy())

            # to print loss
            b = torch.bernoulli(torch.Tensor(conf_val.size()).uniform_(0, 1)).to(device)
            conf_val = conf_val * b + (1 - b)
            pred_new = pred_val * conf_val.expand_as(pred_val) + y_onehot * (1 - conf_val.expand_as(y_onehot))
            pred_new = torch.log(pred_new)

            xentropy_loss = prediction_criterion(pred_new, y)
            confidence_loss = torch.mean(-torch.log(conf))
            total_loss = xentropy_loss + (lmbda * confidence_loss)

            ##############
            # autoencoder
            ##############
            recon_loss = torch.sqrt((x_hat - x).pow(2).mean())
            class_loss = total_loss
            val_loss = recon_loss + args.alpha * class_loss

            # Keep track of metrics
            # losses.update(loss.item())
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i_batch % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {recon_loss:.4f} {class_loss:.4f}\t'.format(i_batch, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      recon_loss=recon_loss,
                                                                        class_loss=class_loss))

    correct = np.array(correct).astype(bool)
    confidence = np.array(confidence)

    plot_histograms(correct, confidence)

    return val_loss, recon_loss, class_loss


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
    recon_loss_graph = []
    class_loss_graph = []

    # Epochs
    for epoch in range(args.epochs):
        # classification network
        cc = 0.
        t = 0.

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train_loss, lmbda_train = train(epoch, train_loader, model, optimizer,
                                        num_classes=args.num_class, correct_count=cc, total=t)

        # One epoch's validation
        val_loss, recon_loss, class_loss = valid(val_loader, model, lmbda_train)
        print('\n * RECON LOSS - {loss:.3f}\n'.format(loss=recon_loss))
        print('\n * CLASS LOSS - {loss:.3f}\n'.format(loss=class_loss))
        print('\n * TOTAL VAL LOSS - {loss:.3f}\n'.format(loss=val_loss))

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
        train_loss_graph.append(train_loss.detach().cpu())
        val_loss_graph.append(val_loss.cpu())
        recon_loss_graph.append(recon_loss.cpu())
        class_loss_graph.append(class_loss.cpu())

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, val_loss, is_best)

    # save train val loss graph
    plt.figure()
    plt.title('(Train, Valid) Loss')
    plt.plot(np.array(epoch_graph), np.array(train_loss_graph), label='train loss')
    plt.plot(np.array(epoch_graph), np.array(val_loss_graph), label='valid loss')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('loss_train_valid.png')

    # save val recon class loss graph
    plt.figure()
    plt.title('(Total, Rcon, Class) Loss')
    plt.plot(np.array(epoch_graph), np.array(val_loss_graph), label='total loss')
    plt.plot(np.array(epoch_graph), np.array(recon_loss_graph), label='recon loss')
    plt.plot(np.array(epoch_graph), np.array(class_loss_graph), label='class loss')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('loss_total_recon_class.png')

if __name__ == '__main__':
    main()
