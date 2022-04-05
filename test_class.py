import argparse
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F


from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader
from models import SegNet
from sklearn import metrics


parser = argparse.ArgumentParser()
parser.add_argument('--major-test-data-path', '-mjr', type=str, default='data/cifar-10/cifar10_mjr_test',
                    required=True, help='major test data path: data/cifar-10/cifar10_mjr_test')
parser.add_argument('--minor-test-data-path', '-mir', type=str, default='data/cifar-10/cifar10_mir_test',
                    required=True, help='minor test data path: data/cifar-10/cifar10_mir_test')
parser.add_argument('--ood-test-data-path', '-ood', type=str, default='data/cifar-10/cifar10_ood_test',
                    required=True, help='ood test data path: data/cifar-10/cifar10_ood_test')
parser.add_argument('--checkpoint', '-ch', type=str, required=True,
                    help='checkpoint path. models/BEST_checkpoint_~.pt')
parser.add_argument('--noise_level', '-n', type=float, default=0.15, required=True,
                    help='noise injection level used in training. default=0.15')
parser.add_argument('--num_class', '-c', type=int, default=10, required=True,
                    help='the number of classes to be classified used in training')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tpr95(ind_confidences, ood_confidences):
    #calculate the falsepositive error when tpr is 95%
    Y1 = ood_confidences
    X1 = ind_confidences

    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / 100000

    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1

    fprBase = fpr / total

    return fprBase


def detection(ind_confidences, ood_confidences, n_iter=100000, return_data=False):
    # calculate the minimum detection error
    Y1 = ood_confidences
    X1 = ind_confidences

    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / n_iter

    best_error = 1.0
    best_delta = None
    all_thresholds = []
    all_errors = []
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        detection_error = (tpr + error2) / 2.0

        if return_data:
            all_thresholds.append(delta)
            all_errors.append(detection_error)

        if detection_error < best_error:
            best_error = np.minimum(best_error, detection_error)
            best_delta = delta

    if return_data:
        return best_error, best_delta, all_errors, all_thresholds
    else:
        return best_error, best_delta


def test(dataloader):
    model = SegNet(in_channels=3, is_unpooling=True, noise_level=args.noise_level, num_classes=args.num_class)
    pretrained_dict = torch.load(args.checkpoint)
    model.load_state_dict(pretrained_dict)
    model = model.to(device)
    model.eval()

    # to print accuracy output, initialize
    correct_list = []
    confidence_list = []

    # to print confidence output, initialize
    out = []

    with torch.no_grad():
        for i_batch, (x, y) in enumerate(dataloader):
            # Set device options
            x = x.to(device)
            y = y.to(device)

            x_hat, classify_out, confidence_out = model(x)

            # confidence part
            confidence = torch.sigmoid(confidence_out)
            confidence = confidence.data.cpu().numpy()
            out.append(confidence)

            # classification part
            pred = F.softmax(classify_out, dim=-1)
            _, pred = torch.max(pred.data, 1)
            correct_list.extend((pred == y).cpu().numpy())

        # classification part
        correct_array = np.array(correct_list).astype(bool)
        accuracy = np.mean(correct_array)

    # confidence part
    out = np.concatenate(out)
    return out, accuracy


def main():
    mjr_test_data = ImageFolder(root=args.major_test_data_path, transform=T.ToTensor())
    mjr_loader = DataLoader(dataset=mjr_test_data, shuffle=False, pin_memory=True)
    mir_test_data = ImageFolder(root=args.minor_test_data_path, transform=T.ToTensor())
    mir_loader = DataLoader(dataset=mir_test_data, shuffle=False, pin_memory=True)
    ood_test_data = ImageFolder(root=args.ood_test_data_path, transform=T.ToTensor())
    ood_loader = DataLoader(dataset=ood_test_data, shuffle=False, pin_memory=True)

    mjr_scores, mjr_acc = test(mjr_loader)
    mjr_labels = np.ones(mjr_scores.shape[0])

    mir_scores, mir_acc = test(mir_loader)
    mir_labels = np.ones(mir_scores.shape[0])

    ood_scores, ood_acc = test(ood_loader)
    ood_labels = np.zeros(ood_scores.shape[0])

    labels = np.concatenate([mjr_labels, mir_labels, ood_labels])
    scores = np.concatenate([mjr_scores, mir_scores, ood_scores])

    ind_scores = np.concatenate([mjr_scores, mir_scores])

    fpr_at_95_tpr = tpr95(ind_scores, ood_scores)
    detection_error, best_delta = detection(ind_scores, ood_scores)
    auroc = metrics.roc_auc_score(labels, scores)
    aupr_in = metrics.average_precision_score(labels, scores)
    aupr_out = metrics.average_precision_score(-1 * labels + 1, 1 - scores)

    # save results to txt file
    f = open('results/[test_class]_' + args.checkpoint[23:-3] + '.txt', 'w')
    f.write("Major accuracy: " + repr(mjr_acc*100) + "\n")
    f.write("Minor accuracy: " + repr(mir_acc*100) + "\n")
    f.write("OOD accuracy: " + repr(ood_acc*100) + "\n")
    f.write("TPR95 (lower is better): " + repr(fpr_at_95_tpr) + "\n")
    f.write("Detection error (lower is better): " + repr(detection_error) + "\n")
    f.write("Best threshold:" + repr(best_delta) + "\n")
    f.write("AUROC (higher is better): " + repr(auroc) + "\n")
    f.write("AUPR_IN (higher is better): " + repr(aupr_in) + "\n")
    f.write("AUPR_OUT (higher is better): " + repr(aupr_out) + "\n")
    f.close()

    # print results
    print("")
    print("TPR95 (lower is better): ", fpr_at_95_tpr)
    print("Detection error (lower is better): ", detection_error)
    print("Best threshold:", best_delta)
    print("AUROC (higher is better): ", auroc)
    print("AUPR_IN (higher is better): ", aupr_in)
    print("AUPR_OUT (higher is better): ", aupr_out)

    # save confidence graph
    ranges = (np.min(scores), np.max(scores))
    plt.figure()
    sns.distplot(mjr_scores.ravel(), hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='Major')
    sns.distplot(mir_scores.ravel(), hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='Minor')
    sns.distplot(ood_scores.ravel(), hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='OOD')
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('results/[test_class]_' + args.checkpoint[23:-3] + '.png')


if __name__ == '__main__':
    main()
