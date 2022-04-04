import os
import random

import cv2 as cv
import numpy as np
import torch

from config import device, save_folder, imsize
from utils import ensure_folder

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--major-test-data-path', '-mjrtest', type=str, default='data/cifar-10/cifar10_mjr_test',
                    required=True, help='major test data path: data/cifar-10/cifar10_mjr_test')
parser.add_argument('--minor-test-data-path', '-mirtest', type=str, default='data/cifar-10/cifar10_mir_test',
                    required=True, help='minor test data path: data/cifar-10/cifar10_mir_test')
parser.add_argument('--ood-test-data-path', '-oodtest', type=str, default='data/cifar-10/cifar10_ood_test',
                    required=True, help='ood test data path: data/cifar-10/cifar10_ood_test')


args = parser.parse_args()


def main():
    mjr_test_data = ImageFolder(root=args.major_test_data_path, transform=T.ToTensor())
    mjr_test_loader = DataLoader(dataset=mjr_test_data, shuffle=False, pin_memory=True)
    mir_test_data = ImageFolder(root=args.minor_test_data_path, transform=T.ToTensor())
    mir_test_loader = DataLoader(dataset=mir_test_data, shuffle=False, pin_memory=True)
    ood_test_data = ImageFolder(root=args.ood_test_data_path, transform=T.ToTensor())
    ood_test_loader = DataLoader(dataset=ood_test_data, shuffle=False, pin_memory=True)

    checkpoint = '{}/BEST_checkpoint.tar'.format(save_folder)  # model checkpoint
    print('checkpoint: ' + str(checkpoint))
    # Load model
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    # test_path = 'data/test/'
    # test_images = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.jpg')]
    #
    # num_test_samples = 10
    # samples = random.sample(test_images, num_test_samples)
    #
    # imgs = torch.zeros([num_test_samples, 3, imsize, imsize], dtype=torch.float, device=device)
    #
    ensure_folder('images')
    # for i, path in enumerate(samples):
    #     # Read images
    #     img = imread(path)
    #     img = imresize(img, (imsize, imsize))
    #     imsave('images/{}_image.png'.format(i), img)
    #
    #     img = img.transpose(2, 0, 1)
    #     assert img.shape == (3, imsize, imsize)
    #     assert np.max(img) <= 255
    #     img = torch.FloatTensor(img / 255.)
    #     imgs[i] = img
    #
    # imgs = torch.tensor(imgs)

    # with torch.no_grad():
    #     preds = model(imgs)
    #
    # for i in range(num_test_samples):
    #     out = preds[i]
    #     out = out.cpu().numpy()
    #     out = np.transpose(out, (1, 2, 0))
    #     out = out * 255.
    #     out = np.clip(out, 0, 255)
    #     out = out.astype(np.uint8)
    #     out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
    #     cv.imwrite('images/{}_out.png'.format(i), out)

    with torch.no_grad():
        for i_batch, (x, y) in enumerate(mjr_test_loader):
            # Set device options
            x = x.to(device)
            y = y.to(device)

            x_hat = model(x)

            out = x_hat.cpu().numpy()
            # reshape [1, 3, 32, 32] -> [3, 32, 32]
            out = np.squeeze(out, axis=0)
            out = np.transpose(out, (1, 2, 0))
            out = out * 255.
            out = np.clip(out, 0, 255)
            out = out.astype(np.uint8)
            out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
            cv.imwrite('images/{}_out.png'.format(i_batch), out)

            # print psnr
            print(10. * np.log10(((1.0 ** 2) / ((x_hat.cpu().numpy() - x.cpu().numpy()) ** 2).mean())))

if __name__ == '__main__':
    main()
