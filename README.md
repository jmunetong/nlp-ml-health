
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
**** Note: This implementation is still UNDER in progress. Algorithm is not fully functional yet*** \

This neural network is based on Hinton et al. `https://proceedings.neurips.cc/paper/2017/hash/2cad8fa47bbef282badbb8de5374b894-Abstract.html` Dynamic routing capsule neural network. In addition, to this architecture, the goal is to implement a Layer to compute the fast Dynamic Routing Based on Weighted Kernel Density Estimation
https://arxiv.org/abs/1805.10807, which may optimize the computation resources for NLP problems. 
Although the dynamic routing algorithm helps capsules achieve more generalization capacity with few parameters, the disadvantage is the large amount of computational requirements of the capsules during the dynamic routing computation. To address this problem, the framework of weighted kernel density estimation is being implemented.

Models NNC_1, NNC_2, and NNC_3 inherit from a Model_Abstract class found in
`model_abstract.py`. This has been done because future work in this pipeline will 
focus on customizing training steps and adding regularization penalties based on
gradients and Hessian matrix. 

#### Params (`__init__`)

- `input_shape`: (int) shape of one observation
- `n_classes`: (int) number of classes in dataset
- `vocab_size`: (int) length of the vocabulary size for the embedding layer
- `embedding_dim`: (int) embedding output dimension

For NNC_Caps, the parameters for initializing model are detailed below:


## Usage 

- In this section we explain how to use the different regularizer objects. It is assumed that
the package has been installed in your virtual environment.

#### Importing

```python

from sota import ModelJacobianSota, ModelVATSota, ModelWARSota
from rdexcaps import ModelJacobianCaps, ModelVATCaps, ModelWARCaps

```


#### Initializing

```python
model = Inception# This is your original/untrained/uncompiled model
model_reg = ModelJacobianSota(model) # pass in your model to the model wrapper for the regularizer
model = RDEX15
model_reg_caps = ModelJacobianCaps(model)

########## IMPORTANT #####################3
# If working with the Wasserstein Regularizer, you need to pass the pre-defined
# pairwise target class cost matrix M into initialization. This matrix is of
# dimensions (# classes, # classes) and its diagonal is assumed to be 0 out.

# Example:
M = [[0, 3.4, 5.6], 
      [3.4, 0, 2.1], 
      [5.6, 2.1, 0]]

model_reg = ModelWarSota(model, M)


```

#### Training Step
```python
# compile the regularized model
model_reg.compile(optimizer=opt,loss=loss, metrics=['acc'], run_eagerly=True)
model_reg_caps.compile(optimizer=opt,loss=['cross_entropy', 'recon'], metrics=['acc'], run_eagerly=True)

# Train model: you may add your callbacks here
xtrain, ytrain = None 
model_reg.fit(xtrain,ytrain, batch_size=32, epochs=100)
model_reg_caps.fit([xtrain,ytrain], [ytrain, xtrain], batch_size=32, epochs=100)

```

#### Evaluation Step

```python
# Testing model

# As you can see we are accessing the model argument and calling predict on 
# your original model, not the regularized model for the evaluation step.
xtest,ytest = None
y_pred = model_reg.model.predict(xtest)
y_pred = model_reg_caps.model.predict(xtest)
```

#### Saving Model

```python
# Saving Model
trained_model = model_reg.save_weights(
    filepath, overwrite=True, save_format=None, options=None
)

```

#### Loading Model

```python
# Loading model: Do not Instantiate the regularizer, just instantiate your
# original model (SOTA, Hybrid) and load its weights
model = DenseNet121()
model_caps = rdex15()

model.load_weights(filepath)
model_caps.load_weights(filepath)

```


## IMPORTANT NOTES! 

Since it might be useful to save the weights as the model is being trained,
beaware that the callback for saving the weights has an argument `save_weights_only`.
This argument must be set to `True` if training models with regularizers. Tensorflow 
will not allow you to save a subclassed keras Model. This only works for functional
or Sequential models.

```python

checkpoint = callbacks.ModelCheckpoint(
        filepath=self.weights_path, 
        monitor=self.watch, 
        verbose=0,
        save_best_only=True, 
        save_weights_only= True
        )
```



## References:
[1] https://arxiv.org/abs/1908.02729
[2] https://arxiv.org/abs/1704.03976
[3] https://arxiv.org/abs/1904.03936
