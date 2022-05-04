import time
import visdom
import argparse
import datetime

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as T

from models import SegNet
from utils import *

import seaborn as sns # import this after torch or it will break everything

d_today = datetime.date.today()
now = datetime.datetime.now()
nowTime = now.strftime('%H:%M')
d_today = str(d_today)[2:4] + str(d_today)[5:7] + str(d_today)[8:10] + '_' + str(nowTime)

parser = argparse.ArgumentParser()
parser.add_argument('--train-data-path', '-train', type=str, default='data/cifar-10/cifar10_train', required=True,
                    help='training data path: data/cifar-10/cifar10_train')
parser.add_argument('--valid-data-path', '-valid', type=str, default='data/cifar-10/cifar10_train', required=True,
                    help='valid data path: data/cifar-10/cifar10_train')
parser.add_argument('--major-test-data-path', '-mjr', type=str, default='data/cifar-10/cifar10_mjr_test',
                    required=True, help='major test data path: data/cifar-10/cifar10_mjr_test')
parser.add_argument('--minor-test-data-path', '-mir', type=str, default='data/cifar-10/cifar10_mir_test',
                    required=True, help='minor test data path: data/cifar-10/cifar10_mir_test')
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
parser.add_argument('--resample', '-r', type=float, default=3, required=False,
                    help='Resampling ratio. if it is 3, then resample minor samples 3 times more than major samples')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    plt.savefig('results/[train]maxloss_full_b_' + str(args.batch_size) + '_n_' + str(args.noise_level)
                + '_a_' + str(args.alpha) + '_c_' + str(args.num_class) + '_accuracy_conf_' + d_today + '.png')


def train(epoch, train_loader, model, optimizer, correct_count, total):
    model.train()

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    start = time.time()

    # for classifier
    lmbda = 0.1

    # loss_record_list
    loss_record_list = []

    max_loss = 0.
    # Batches
    for i_batch, (x, y) in enumerate(train_loader):
        # Set device options
        x = x.to(device)
        y = y.to(device)
        y_onehot = encode_onehot(y, args.num_class)

        # Zero gradients
        optimizer.zero_grad()

        # model output
        _, x_hat, classify, confidence = model(x)

        #############
        # Classifier
        #############
        # classify output, confidence output
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

        # classifier total loss
        class_loss = xentropy_loss + (lmbda * confidence_loss)

        # 0.3 = default hint budget
        if 0.3 > confidence_loss.data:
            lmbda = lmbda / 1.01
        else:
            lmbda = lmbda / 0.99

        # for every batch, calculate loss and update weighted loss
        for i in range(np.array(x.size())[0]):
            # MSE loss
            loss = (x_hat[i, :, :, :] - x[i, :, :, :]).pow(2).mean()
            auto_max_loss = max(loss, max_loss)

    # Final loss: (auto loss) + a * (class loss)
    loss = auto_max_loss + args.alpha * class_loss
    loss.backward()

    # loss_record: to print loss value at the terminal
    loss_record = auto_max_loss + args.alpha * class_loss
    loss_record_list.append(loss_record)

    # classifier
    pred_idx = torch.max(classify.data, 1)[1]
    total += y.size(0)
    correct_count += (pred_idx == y.data).sum()
    accuracy = correct_count / total

    optimizer.step()

    batch_time.update(time.time() - start)
    start = time.time()

    # Print status for every 20 batch
    print('Epoch: [{0}][{1}/{2}]\t'
          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Loss {loss:.5f}\t'
          'Class accuracy {accuracy:.3f}\t'
          .format(epoch, i_batch, len(train_loader), batch_time=batch_time,
                  loss=loss_record, accuracy=accuracy*100))

    loss_record = sum(loss_record_list)/len(loss_record_list)

    return loss_record, lmbda


def valid(val_loader, model, lmbda):
    model.eval()

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    start = time.time()

    max_loss = 0.

    # for classifier
    correct = []
    confidence = []

    # to draw confidence graph
    out = []

    with torch.no_grad():
        for i_batch, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)
            y_onehot = encode_onehot(y, args.num_class)

            _, x_hat, pred_out, conf_output = model(x)

            #############
            # classifier
            #############
            pred_val = F.softmax(pred_out, dim=-1)
            conf = torch.sigmoid(conf_output).data.view(-1)
            conf_val = torch.sigmoid(conf_output)

            pred_value, pred = torch.max(pred_val.data, 1)
            correct.extend((pred == y).cpu().numpy())
            confidence.extend(conf.cpu().numpy())

            # to draw confidence graph
            pred_c = conf_val.data.cpu().numpy()
            out.append(pred_c)

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
            # for every batch, calculate loss and update weighted loss
            for i in range(np.array(x.size())[0]):
                # MSE loss
                loss = (x_hat[i, :, :, :] - x[i, :, :, :]).pow(2).mean()
                auto_max_loss = max(loss, max_loss)

            class_loss = total_loss
            val_loss = auto_max_loss + args.alpha * class_loss

            batch_time.update(time.time() - start)
            start = time.time()

            # Print status for every 20 batch
            if i_batch % 20 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {recon_loss:.4f} {class_loss:.4f}\t'.format(i_batch, len(val_loader),
                                                                        batch_time=batch_time,
                                                                        recon_loss=auto_max_loss,
                                                                        class_loss=class_loss))

    correct = np.array(correct).astype(bool)
    confidence = np.array(confidence)

    plot_histograms(correct, confidence)

    # to draw confidence graph
    out = np.concatenate(out)

    return val_loss, auto_max_loss, class_loss, out


def main():
    train_data = ImageFolder(root=args.train_data_path, transform=T.ToTensor())
    if args.resample is not None:
        print('+++++ Resampling mode +++++')
        sample_weights = np.ones([len(train_data), ])
        for i, (img, label) in enumerate(train_data.imgs):
            if label == 0 or label == 2:
                sample_weights[i] = args.resample
        train_sampler = data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    else:
        train_sampler = data.RandomSampler(train_data)

    # Because we use sampler, 'shuffle' in 'train_loader' must be False.
    train_loader = DataLoader(dataset=train_data, sampler=train_sampler, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True)
    valid_data = ImageFolder(root=args.valid_data_path, transform=T.ToTensor())
    val_loader = DataLoader(dataset=valid_data, batch_size=len(valid_data), shuffle=False, pin_memory=True)

    # to draw confidence graph
    mjr_test_data = ImageFolder(root=args.major_test_data_path, transform=T.ToTensor())
    mjr_loader = DataLoader(dataset=mjr_test_data, shuffle=False, pin_memory=True)
    mir_test_data = ImageFolder(root=args.minor_test_data_path, transform=T.ToTensor())
    mir_loader = DataLoader(dataset=mir_test_data, shuffle=False, pin_memory=True)

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
        if epochs_since_improvement == 40:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 10 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train_loss, lmbda_train = train(epoch, train_loader, model, optimizer, correct_count=cc, total=t)

        # One epoch's validation
        val_loss, recon_loss, class_loss, _ = valid(val_loader, model, lmbda_train)
        print('\n * RECON LOSS - {loss:.3f}\n'.format(loss=recon_loss))
        print('\n * CLASS LOSS - {loss:.3f}\n'.format(loss=class_loss))
        print('\n * TOTAL VAL LOSS - {loss:.3f}\n'.format(loss=val_loss))

        # Check if there was an improvement
        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % epochs_since_improvement)
        else:
            epochs_since_improvement = 0

        # draw loss graph
        epoch_graph.append(epoch + 1)
        train_loss_graph.append(train_loss.detach().cpu())
        val_loss_graph.append(val_loss.cpu())
        recon_loss_graph.append(recon_loss.cpu())
        class_loss_graph.append(class_loss.cpu())

        # Save checkpoint
        if is_best:
            torch.save(model.state_dict(), 'models/BEST_checkpoint_maxloss_full' + '_b_' + str(args.batch_size)
                       + '_n_' + str(args.noise_level) + '_a_' + str(args.alpha)
                       + '_c_' + str(args.num_class) + '_' + d_today + '.pt')
            # draw confidence graph
            _, _, _, mjr_scores = valid(mjr_loader, model, lmbda_train)
            _, _, _, mir_scores = valid(mir_loader, model, lmbda_train)
            scores = np.concatenate([mjr_scores, mir_scores])
            ranges = (np.min(scores), np.max(scores))
            plt.figure()
            sns.distplot(mjr_scores.ravel(), hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='Major')
            sns.distplot(mir_scores.ravel(), hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='Minor')
            plt.xlabel('Confidence')
            plt.ylabel('Density')
            plt.legend()
            plt.savefig('results/[train]maxloss_full_b_' + str(args.batch_size) + '_n_' + str(args.noise_level)
                + '_a_' + str(args.alpha) + '_c_' + str(args.num_class) + '_confidence_figure_' + d_today + '.png')

    # save train val loss graph
    plt.figure()
    plt.title('(Train, Valid) Loss')
    plt.plot(np.array(epoch_graph), np.array(train_loss_graph), label='train loss')
    plt.plot(np.array(epoch_graph), np.array(val_loss_graph), label='valid loss')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('results/[train]maxloss_full_b_' + str(args.batch_size) + '_n_' + str(args.noise_level)
                + '_a_' + str(args.alpha) + '_c_' + str(args.num_class) + '_loss_train_valid_' + d_today + '.png')

    # save val recon class loss graph
    plt.figure()
    plt.title('(Total, Rcon, Class) Loss')
    plt.plot(np.array(epoch_graph), np.array(val_loss_graph), label='total loss')
    plt.plot(np.array(epoch_graph), np.array(recon_loss_graph), label='recon loss')
    plt.plot(np.array(epoch_graph), np.array(class_loss_graph), label='class loss')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('results/[train]maxloss_full_b_' + str(args.batch_size) + '_n_' + str(args.noise_level)
                + '_a_' + str(args.alpha) + '_c_' + str(args.num_class) + '_loss_total_recon_class_' + d_today + '.png')


if __name__ == '__main__':
    main()
