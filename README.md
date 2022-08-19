
# ML SYSTEM VERIFY HEALTH-CLAIMS FACT-CHECK

## Description

This module/package provides a machine learning pipeline for training and querying claims from the pub-health dataset, which contains valid, false, mixed, and unknown claims. 
Here, you will find 4 deep neural networks that have been customized:

- ### NNC_1
- ### NNC_2
- ### NNC_3
- ### NNC_Caps


## Table of Contents 

- [Overview](#overview)
- [Installation](#installation)
- [Models](#objects)
- [Usage](#usage)
- [License](#license)
- [References](#references)

## Overview

### Why Model Regularization is Important? 

In the field of machine learning, a reliable model is one that can withstand 
shifts from training data to unseen test data. When this pattern is present
in model classification, we know the model is able to generalize. However, overfitting
to training data can be a common behavior in classifiers, especially in complex 
nonlinear systems like deep learning models. Any instability in a system like a 
trained neural network can further be exploited by adversaries or data drift presence
to render these model useless. It is, therefore, important to ensure that models
be stable against perturbations or slight shifts in data distribution. Regularization
is useful in this issue becuase it introduces additional information in order 
to manage this inevitable gap between the training error and the test error.

### Jacobian Regularizer

The Jacobian Matrix provides crucial information to understand the extent to 
which a model prediction is unstable with respect to input perturbations: 
the larger the components of the Jacobian matrix, then the more unstable the 
prediction is. Thus, a keen approach to increase stability performance in a 
model’s prediction would be to minimize the square of the Frobenius norm of 
the input-output Jacobian matrix.

The Jacobian regularization term brings a convenient advantage in the training 
step of neural networks. With the current neural network training packages, 
such as Tensorflow and Pytorch, including this regularization does not 
significantly interfere with the training steps. Concretely, it is possible 
to add the jacobian regularizer to the loss function of any model and perform 
training optimization using Stochastic Gradient Descent. [1]

<img src="/imgs/jacobian.png" alt="Alt text" title="Optional title">

### Virtual Adversarial Training Regularizer

The Virtual Adversarial Training Regularizer (VAT) is a method that trains 
the output distribution of a classifier to be isotropically smooth around each input data point 
by smoothing the model in its most anisotropid direction. It creates adversarials
that are defined as perturbations that can alter the output distribution in
the sense of distributional divergence. Namely, the virtual adversarial direction
is the most anisotropic direction. In contrast with previous work on adversarial
training, this type of regularizer uses a virtual adversarial direction defined
on an unlabeled data point: it is the direction that can most greatly deviate 
the current inferred output distribution from the status quo.
In other words, even in the absence of label information, virtual adversarial
direction can be defined on an unlabeled data point as if there is a 
“virtual” label; hence the name “virtual” adversarial direction.

The local anisotropy of a classifier can be computed at each input point without
using supervised signal.This type of regularizer computes the local distributional
smoothness (LDS), which is defined to be the divergence-based distributional 
robustness of the model against virtual adversarial direction. This type of regularizer
is an afficient method that can be used during model training; it maximizes the 
liklihood of the model while promoting the model's LDS on each training input data.[2]

<img src="/imgs/vat.png" alt="Alt text" title="Optional title">

### Wasserstein Adversarial Training Regularizer

The Wasserstein Adversarial Regularization (WAR) model follows a similar approach to that
of the VAT regularizer. This time, howver, the WAR penalizes a classifier
according to similarity between classes: in doing so, it allows for learning complex
class boundaries between classes that are too similar to each other, while having
simpler class boundaries between classes showing low similarity. This type of regularization
also reduces the discrepancy between the prediction of a true input and the created, 
unlabeled near-by adversarial sample. As a cost measure, this regularizer uses
a loss based on the Wasserstein distance, which is computed with respect to a 
ground cost encoding of between-class similarities. This ground cost provides 
the flexibility to regularize with different strengths pairs of classes. While
computing the Wasserstein Loss can be costly, especially in high-dimensional 
data, the WAR regularizer approximates this optimal-transport problem with the 
Sinkhorn Loss. [3]

<img src="/imgs/war.png" alt="Alt text" title="Optional title">
<img src="/imgs/cost_war.png" alt="Alt text" title="Optional title">


## Installation

Run the following command in your terminal with your virtual environment. If 
using conda, make sure you have pip installed. 

```pip install git+https://gitlab.com/r-dex-research-group/r-dex-regularizers.git```

Otherwise, download repository and pay attention to `requirements.txt`.

## Objects

### Jacobian Regularizer

#### Hybrid: `rdexcaps.ModelJacobian`

#### SOTA: `sota.ModelJacobian`


This applies to both Hybrid and Single Output models.

#### Params (`__init__`)
- `model`: (tf.keras.model) model used for training 
- `input_shape`: tuple(ints) [height, width, channels], shape of the data
- `n_classes`: (int) number of classes in dataset
- `lambda_JR`: (float) importance parameter for jacobian regularization. Default 0.01
- `n_proj`: (int) number of projections. It should also
            be greater than 0 and less than sqrt(# of classes). We can also set 
            it to be n_proj=-1 to compute the full jacobian which is computa-tionally
            expensive. Default 10
- `spectral`: (bool, optional)
            whether creating random projections or doing one pass computation 
            with jacobian. True creates random projections. Default True
- `**kwargs`: (optional)

### Virtual Adversarial Regularizer

#### Hybrid: `rdexcaps.ModelVAT`

#### SOTA: `sota.ModelVAT`

This applies to both Hybrid and Single Output models.

### Params (`__init__`)
- `model`: (tf.keras.model) model used for training 
- `input_shape`: tuple(ints) [height, width, channels], shape of the data
- `n_classes`: (int) number of classes in dataset
- `eps`: (float) epsilon constraint upper bound value for the amount of perturbation
            allowed. Default 0.05
- `n_iter`: (int) 
            number of iterations for identifying direction vector. Default 1
- `alpha`: (float)
            weight of VAT regularization term that will be added to classifier's
            loss function. Default 0.05
- `xi`: float (optional)
            normalization constraint for normalizing d vector. Default 1e-6
- `**kwargs`: (optional)


### Wasserstein Adversarial Regularizer

#### Hybrid: `rdexcaps.ModelWAR`

#### SOTA: `sota.ModelWAR`

This applies to both Hybrid and Single Output models.

### Params (`__init__`)
- `model`: (tf.keras.model) model used for training 
- `M`: (np.ndarray or tf.tensor) (dimensions: (n_classes, n_classes)
        matrix that contained the pre-defined pair-wise class cost from moving 
        distributional properties from one class to the other. 
        The ground cost C reflects the geometry of the label space. One approach for 
        computing the ground cost is to use the prior labels (the one-hot-encoded)
        training labels from the training dataset used, and add noise to them 
        examples include label smoothing.  Other approaches
        include using a variational autoencoder for embeddings on the data and
        then compute the class-wise euclidean distance. IMPORTANT: It is assume that the diagonal of this 
        matrix is composed of zeros!!!!
- `input_shape`: tuple(ints) [height, width, channels], shape of the data
- `n_classes`: (int) number of classes in dataset
- `eps`: (float) 
        epsilon constraint upper bound value for the amount of perturbation
        allowed. Default 2.5
- `n_iter`: (int) 
         number of iterations for computing d perturbation vector in method __war_loss
         Default 1.  
- `alpha`: (float)
        importance value added in the total loss for the WAR regularizer in 
        the classifier's loss function prior to computing the gradients. Default
        1.0
- `xi`: float (optional)
        normalization constraint for normalizing d vector. Default 1e-6
- `nb_loops`: int (optional)
        number of iterations to compute P in __sikhorn loss method. Default
        5
- `lambda_reg`: float (optional)
        Entropic Regularization term for Sinkhorn loss. Default 0.05.
        Note: Do not make this number too small. It might send sinkhorn loss to
        nan
- `**kwargs`: (optional)


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
