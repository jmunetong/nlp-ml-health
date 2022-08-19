
# Machine Learning System for Claim Validity

## Description

This module/package provides a machine learning pipeline for training and querying claims from the pub-health dataset, which contains valid, false, mixed, and unknown claims. 
In this pipeline, we implemented 4 customized neural networks for natural language processing, all of which use an embedding layer. Because of limited computing resources for training,transformers from the HuggingFace were not used for these experiments

- ##### NNC_1
- ##### NNC_2
- ##### NNC_3
- ##### NNC_Caps


## Table of Contents 

- [Overview](#overview)
- [Installation](#installation)
- [Models](#objects)
- [Usage](#usage)
- [License](#license)
- [References](#references)


## Overview
This module contains a `config.py` file that encompasses the main parameters necessary
for data-tokenizing and model running. Feel free to modify any of these values in
the dictionary to customize model output.
Each experiment outputs a directory with a unique name and id value stored in `/results` directory. Here, you will find a summary of the experiment performance, architecture of the model, weights as an `.hd5` file, and a `.json` file containing the parameters used for the experiment. 
At the end of an experiment, accuracy and loss will be printed.

## Installation

Download repository and use `requirements.txt` file for venv package installation

## Objects

The algorithms used in this pipeline can be found in `/models` folder. Note, each
of these algorithms contains a UnitTest.

##### NNC_1
A small natural-language based neural network containing an embedding layer for
input-id mappings. As a baseline model, it only contains one convolution. Data
is thenprojected into a GlobalMaxpooling layer, and a classifier block.

##### NNC_2
An extension model from NNC_1: It includes 4 convolutions with MaxPooling layers
within each other

##### NNC_3
NNC_3 is also a convolutional-based algorithm with an embedding layer for input_id mappings. This algorithm has a hierarchical based convolutional structure with causal layers that allow for dilation and an enlargement of the receptive fields as it goes 
deeper in the convolutions. The output of each convolution is passed to a set 
of attention weights to selectively increase weights to crucial data transformation
or decrease otherwise in irrelevant information. This form of architecture has been 
based on `https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9413635`
##### NNC_Caps
**** Note: This implementation is still UNDER in progress. Algorithm is not fully functional yet*** 

This algorithm is based on a dynamic routing capsule neural network. In addition, to this architecture, the goal is to implement a Layer to compute the fast Dynamic Routing Based on Weighted Kernel Density Estimation
https://arxiv.org/abs/1805.10807, which may optimize the computation resources for NLP problems. 
Although the dynamic routing algorithm helps capsules achieve more generalization capacity with few parameters, the disadvantage is the large amount of computational requirements of the capsules during the dynamic routing computation. To address this problem, the framework of weighted kernel density estimation is being implemented.

Models NNC_1, NNC_2, and NNC_3 inherit from a Model_Abstract class found in
`model_abstract.py`. This has been done because future work in this pipeline will 
focus on customizing training steps and adding regularization penalties based on
gradients and Hessian matrix. 

This neural network is based on Hinton et al. `https://proceedings.neurips.cc/paper/2017/hash/2cad8fa47bbef282badbb8de5374b894-Abstract.html`

#### Params (`__init__`)

- `input_shape (int)`: shape of one observation
- `n_classes (int)`: number of classes in dataset
- `vocab_size (int)`: length of the vocabulary size for the embedding layer
- `embedding_dim (int)`: embedding output dimension

For NNC_Caps, the parameters for initializing model are detailed below:

#### Params (`__init__`)
- `vocab_size (int)`: size of vocabulary
- `num_classes (int)`: number of classes in the dataset
- `dim_capsule (int)`: length of capsule of neural network
- `num_compressed_capsule` (int): reduction of length vector for memory capacity
- `embedding_dim (int)`: length of embedding vector for input_id mapping
- `max_length (int)`: maximum sentence length from input_id


## Usage 

- To run an experiment, modify configs.py if in need of customizing training hyperparameters or loss function, etc. To run one experiment use the `main.py`
file. As follows:

```
python3 main.py --model <"model_name">
```
Candidates for model names are `nnc_1`,`nnc_2`, and `nnc_3`. Note `nnc_caps`
is in current development.

Additionaly, you may find the `run.sh` file helpful to run a series of experiments
for each of the models. Just type in terminal command:

```
sh run.sh
```

