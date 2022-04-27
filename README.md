# Autoencoder

This repository is to do convolutional autoencoder with SetNet based on Cars Dataset from Stanford.


## Dataset
Cifar 10 dataset
* Major dataset: class 3, 4, 5, 6, 7
* Minor dataset: class 0, 2
* OOD dataset: class 1, 8, 9

## Architecture

![image](https://github.com/foamliu/Conv-Autoencoder/raw/master/images/segnet.jpg)

## Usage


### Train (ver1. minibatch max loss)
```bash
$  CUDA_VISIBLE_DEVICES=0 python train_maxloss_minibatch.py -train 'data/cifar-10/cifar10_train' -valid 'data/cifar-10/cifar10_valid' -mjr 'data/cifar-10/cifar10_mjr_test' -mir 'data/cifar-10/cifar10_mir_test' -b 64 -lr 0.0001 -e 300 -n 0.15 -c 10 -a 0.2
```
### Train (ver2. stochastic weighted loss)
```bash
$  CUDA_VISIBLE_DEVICES=0 python train_weighted_loss.py -train 'data/cifar-10/cifar10_train' -valid 'data/cifar-10/cifar10_valid' -mjr 'data/cifar-10/cifar10_mjr_test' -mir 'data/cifar-10/cifar10_mir_test' -b 64 -lr 0.0001 -e 300 -n 0.15 -c 10 -a 0.2
```
### Test Autoencoder
```bash
$ python test_auto.py -mjr 'data/cifar-10/cifar10_mjr_test' -mir 'data/cifar-10/cifar10_mir_test' -ood 'data/cifar-10/cifar10_ood_test' -ch 'models/BEST_checkpoint_b_64_n_0.15_a_0.2.pt' -n 0.15 -c 10
```
### Test Classifier
```bash
$ python test_class.py -mjr 'data/cifar-10/cifar10_mjr_test' -mir 'data/cifar-10/cifar10_mir_test' -ood 'data/cifar-10/cifar10_ood_test' -ch 'models/BEST_checkpoint_b_64_n_0.15_a_0.2.pt' -n 0.15 -c 10
```
## Folders
* **data**
  * put dataset here
* **images**
  * reconstructed images (outputs of *test_auto.py*)
* **models**
  * best checkpoint (outputs of *train.py*)
* **results**
  * [train] outputs from *train.py*
  * [test_auto] outputs from *test_auto.py*
  * [test_class] outputs from *test_class.py*