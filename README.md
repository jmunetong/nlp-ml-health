
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

##### NNC_1

##### NNC_2
##### NNC_3
##### NNC_Caps


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
