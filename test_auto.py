import cv2
import numpy as np
import torch

from utils import ensure_folder

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader
import argparse
from models import SegNet


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


def main():
    mjr_data = ImageFolder(root=args.major_test_data_path, transform=T.ToTensor())
    mjr_loader = DataLoader(dataset=mjr_data, shuffle=False, pin_memory=True)
    mir_data = ImageFolder(root=args.minor_test_data_path, transform=T.ToTensor())
    mir_loader = DataLoader(dataset=mir_data, shuffle=False, pin_memory=True)
    ood_data = ImageFolder(root=args.ood_test_data_path, transform=T.ToTensor())
    ood_loader = DataLoader(dataset=ood_data, shuffle=False, pin_memory=True)

    model = SegNet(in_channels=3, is_unpooling=True, noise_level=args.noise_level, num_classes=args.num_class)
    pretrained_dict = torch.load(args.checkpoint)
    model.load_state_dict(pretrained_dict)
    model = model.to(device)
    model.eval()

    ensure_folder('images')
    ensure_folder('results')

    # save results to txt file
    f = open('results/[test_auto]_' + args.checkpoint[23:-3] + '.txt', 'w')

    # repeat for 3 states: mjr, mir, ood
    for i in range(3):
        if i == 0:
            dataloader = mjr_loader
            psnr = []
            mode = 'major'
        elif i == 1:
            dataloader = mir_loader
            psnr = []
            mode = 'minor'
        else:
            dataloader = ood_loader
            psnr = []
            mode = 'ood'

        with torch.no_grad():
            for i_batch, (x, y) in enumerate(dataloader):
                # Set device options
                x = x.to(device)
                y = y.to(device)

                x_hat, _, _ = model(x)

                out = x_hat.cpu().numpy()
                # reshape [1, 3, 32, 32] -> [3, 32, 32]
                out = np.squeeze(out, axis=0)
                out = np.transpose(out, (1, 2, 0))
                out = out * 255.
                out = np.clip(out, 0, 255)
                out = out.astype(np.uint8)
                out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                cv2.imwrite('images/' + mode + '_{}_out.png'.format(i_batch), out)
                psnr.append(10. * np.log10(((1.0 ** 2) / ((x_hat.cpu().numpy() - x.cpu().numpy()) ** 2).mean())))

        print(mode, ': ', sum(psnr)/len(psnr), '\n')

        f.write(repr(mode) + "(PSNR):" + repr(sum(psnr)/len(psnr)) + "\n")
    f.close()
        
if __name__ == '__main__':
    main()
