# cs231n: Convolutional Neural Networks for Visual Recognition

Assignments and slides for [cs231n](http://cs231n.stanford.edu/).

Fork this repo, clone your fork, and follow the instructions below for getting started.

## Setup

The course uses Python 2 and `virtualenv` to manage the environment.  Converting the code to python 3 is too much
of a hassle (with little benefit) and so we will use Python 2 but we will use [anaconda](https://www.continuum.io/downloads) to manage the environment.

Once you've cloned this repo run this from the root of the repo to create the environment:

```
conda env create
source activate cs231n
(cd assignment3/cs231n; python setup.py install)
```

### Getting the data

For assignments 1 and 2 you will need the CIFAR10 dataset. To download it run (from the root of the repo):

```
(cd datasets; ./get_cifar10.sh)
```

For assignment 3 you will need to run the other scripts in the `datasets` dir or you can just run
`./get_datasets.sh` to download all of them.

NOTE: For people in Cebu, these are large downloads and so you should only have one person download the datasets
and then you can copy them over your local network or use flash drives.


## How to work on the assignments

Activate the correct python environment with:

```
source activate cs231n
```

Start up the Juptyer (IPython) Notebook:

```
jupyter notebook
```

Each assignment has notebooks that will guide you through editing the various files to complete the
assignment. Refer to each of the assignment pages to start them:


[Assignment #1: Image Classification, kNN, SVM, Softmax, Neural Network](https://cs231n.github.io/assignments2016/assignment1/)

[Assignment #2: Fully-Connected Nets, Batch Normalization, Dropout, Convolutional Nets](https://cs231n.github.io/assignments2016/assignment2/)

[Assignment #3: Recurrent Neural Networks, Image Captioning, Image Gradients, DeepDream](https://cs231n.github.io/assignments2016/assignment3/)
