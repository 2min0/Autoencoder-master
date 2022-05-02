# Autoencoder

This repository is to do convolutional autoencoder with SetNet based on Cars Dataset from Stanford.


## Dataset
Cifar 10 dataset
* Major dataset: class 3, 4, 5, 6, 7
* Minor dataset: class 0, 2
* OOD dataset: class 1, 8, 9

## Architecture

* Autoencoder structure (actually, the last 'Softmax' layer is not needed.)

![image](https://github.com/foamliu/Conv-Autoencoder/raw/master/images/segnet.jpg)
* Autoencoder + Classifier structure
  * can distinguish major, minor, and odd data using both autoencoder output and classifier output.

![image](https://user-images.githubusercontent.com/90391927/166187657-97a23864-f827-4f7f-94a6-056a3022bcb8.png)
## Usage


### Train (ver1. vanilla autoencoder)
```bash
$  CUDA_VISIBLE_DEVICES=0 python train_autoencoder.py -train 'data/cifar-10/cifar10_train' -valid 'data/cifar-10/cifar10_valid' -mjr 'data/cifar-10/cifar10_mjr_test' -mir 'data/cifar-10/cifar10_mir_test' -b 64 -lr 0.0001 -e 300 -n 0.15 -c 10
```
### Train (ver2. batch training, max loss)
* In this mode, save all loss values and only back propagate one max loss per one epoch.
* Therefore, actually the batch size (-b 64) is meaningless.
```bash
$  CUDA_VISIBLE_DEVICES=0 python train_maxloss_batch.py -train 'data/cifar-10/cifar10_train' -valid 'data/cifar-10/cifar10_valid' -mjr 'data/cifar-10/cifar10_mjr_test' -mir 'data/cifar-10/cifar10_mir_test' -b 64 -lr 0.0001 -e 300 -n 0.15 -c 10 -a 0.2
```
### Train (ver3. minibatch training, weighted loss)
* In this mode, save all loss values and set a threshold as '1.5 x average loss'.
* And then, for the losses whose value is greater than the threshold, use '1.5 x loss' instead of 'loss'.
```bash
$  CUDA_VISIBLE_DEVICES=0 python train_weightloss_minibatch.py -train 'data/cifar-10/cifar10_train' -valid 'data/cifar-10/cifar10_valid' -mjr 'data/cifar-10/cifar10_mjr_test' -mir 'data/cifar-10/cifar10_mir_test' -b 64 -lr 0.0001 -e 300 -n 0.15 -c 10 -a 0.2
```
### Train (ver4. stochastic training, weighted loss)
* Same as ver3, but stochastic training.
* Use SGD optimizer instead of Adam optimizer.
```bash
$  CUDA_VISIBLE_DEVICES=0 python train_weightloss_stochastic.py -train 'data/cifar-10/cifar10_train' -valid 'data/cifar-10/cifar10_valid' -mjr 'data/cifar-10/cifar10_mjr_test' -mir 'data/cifar-10/cifar10_mir_test' -b 1 -lr 0.0001 -e 300 -n 0.15 -c 10 -a 0.2
```
### Train (ver5. minibatch training, revised weighted loss)
* Use squared MSE loss to obtain weighted loss effect.
```bash
$  CUDA_VISIBLE_DEVICES=0 python train_weightloss_minibatch_revised.py -train 'data/cifar-10/cifar10_train' -valid 'data/cifar-10/cifar10_valid' -mjr 'data/cifar-10/cifar10_mjr_test' -mir 'data/cifar-10/cifar10_mir_test' -b 1 -lr 0.0001 -e 300 -n 0.15 -c 10 -a 0.2
```
### Train (ver6. minibatch training, revised weihgted loss, resampling)
* Use squared MSE loss to obtain weighted loss effect.
* By using _'-r 3'_, we can sample minor data 3 times more than major data.
```bash
$  CUDA_VISIBLE_DEVICES=0 python train_weightloss_minibatch_revised_resample.py -train 'data/cifar-10/cifar10_train' -valid 'data/cifar-10/cifar10_valid' -mjr 'data/cifar-10/cifar10_mjr_test' -mir 'data/cifar-10/cifar10_mir_test' -b 1 -lr 0.0001 -e 300 -n 0.15 -c 10 -a 0.2 -r 3
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