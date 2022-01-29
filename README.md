# Beta-Divergence Loss Implementations

This repository contains code for Python implementations of the beta-divergence loss, including implementations compatible [NumPy](https://numpy.org/) and [PyTorch](https://pytorch.org/).


## Dependencies

This library is written in Python, and requires Python (with recommended version >= 3.9) to run. In addition to a working Pytorch installation, this library relies on the following libraries and recommended version numbers:

* [Python](https://www.python.org/) >= 3.9
* [NumPy](https://numpy.org/) >= 1.22.0
* [SciPy](https://www.scipy.org/) >= 1.7.3


## Installation

To install the latest stable release, use [pip](https://pip.pypa.io/en/stable/). Use the following command to install:

    $ pip install beta-divergence-metrics


## Usage

The [`numpybd.loss`](https://github.com/wecarsoniv/beta-divergence-metrics/blob/main/src/numpy/loss.py) module contains two beta-divergence function implementations compatible with NumPy and NumPy arrays: one general beta-divergence between two arrays, and a beta-divergence implementation specific to non-negative matrix factorization (NMF). Similarly [`torchbd.loss`](https://github.com/wecarsoniv/beta-divergence-metrics/blob/main/src/torchbd/loss.py) module contains two beta-divergence class implementations compatible with Pytorch and PyTorch tensors. Beta-divergence implementations can be imported as follows:

```python
# Import beta-divergence loss implementations
from numpybd.loss import *
from torchbd.loss import *

```


### Beta-divergence between two NumPy arrays

To calculate the beta-divergence between a NumPy array `a` and a target or reference array `b`, use the `beta_div_loss` loss function. The `beta_div_loss` loss function can be used as follows:

```python
# Calculate beta-divergence loss between array a and target array b
loss_val = beta_div_loss(beta=0, reduction='mean')

```


### Beta-divergence between two PyTorch tensors

To calculate the beta-divergence between tensor `a` and a target or reference tensor `b`, use the `BetaDivLoss` loss function. The `BetaDivLoss` loss function can be instantiated and used as follows:

```python
# Instantiate beta-divergence loss object
beta_div_loss = BetaDivLoss(beta=0, reduction='mean')

# Calculate beta-divergence loss between tensor a and target tensor b
loss_val = beta_div_loss(input=a, target=b)

```


### NMF beta-divergence between NumPy array of data and data reconstruction

To calculate the NMF-specific beta-divergence between a NumPy array of data matrix `X` and the product of a scores matrix `H` and a components matrix `W`, use the `nmf_beta_div_loss` loss function. The `nmf_beta_div_loss` loss function can beused as follows:

```python
# Calculate beta-divergence loss between data matrix X (target or
# reference matrix) and matrix product of H and W
loss_val = nmf_beta_div_loss(X=X, H=H, W=W, beta=0, reduction='mean')

```


### NMF beta-divergence between PyTorch tensor of data and data reconstruction

To calculate the NMF-specific beta-divergence between a PyTorch tensor of data matrix `X` and the matrix product of a scores matrix `H` and a components matrix `W`, use the `NMFBetaDivLoss` loss class function. The `NMFBetaDivLoss` loss function can be instantiated and used as follows:

```python
# Instantiate NMF beta-divergence loss object
nmf_beta_div_loss = NMFBetaDivLoss(beta=0, reduction='mean')

# Calculate beta-divergence loss between data matrix X (target or
# reference matrix) and matrix product of H and W
loss_val = nmf_beta_div_loss(X=X, H=H, W=W)

```


### Choosing beta value

When instantiating beta-divergence loss objects, the value of beta should be chosen depending on data type and application. For NMF applications, a beta value of 0 (Itakura-Saito divergence) is recommemded. Integer values of beta correspond to the following divergences and loss functions:

* beta = 0: [Itakura-Saito divergence](https://en.wikipedia.org/wiki/Itakura-Saito_distance)
* beta = 1: [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback-Leibler_divergence)
* beta = 2: [mean-squared error](https://en.wikipedia.org/wiki/Mean_squared_error)


## Issue Tracking and Reports

Please use the [GitHub issue tracker](https://github.com/wecarsoniv/beta-divergence-metrics/issues) associated with this repository for issue tracking, filing bug reports, and asking general questions about the package or project.

