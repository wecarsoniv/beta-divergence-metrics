# Beta-Divergence Loss - PyTorch Implementations

This repository contains code for the Python package `torchbd`: PyTorch implementations of the beta-divergence loss.


## Dependencies

The [`torchbd`](https://github.com/wecarsoniv/pytorch-beta-divergence/tree/main/src/torchbd) package is written in Python, and requires Python (with recommended version >= 3.9) to run. In addition to a working Pytorch installation, this package relies on the following libraries and version numbers:

* [Python](https://www.python.org/) >= 3.9
* [NumPy](https://numpy.org/) >= 1.22.0
* [SciPy](https://www.scipy.org/) >= 1.7.3


## Installation

To install the latest stable release, use [pip](https://pip.pypa.io/en/stable/). Use the following command to install:

    $ pip install pytorch-beta-divergence


## Usage

The [`loss.py`](https://github.com/wecarsoniv/pytorch-beta-divergence/blob/main/src/torchbd/loss.py) module contains two beta-divergence implementations: one general beta-divergence between two 2-dimensional matrices or tensors, and a beta-divergence implementation specific to non-negative matrix factorization (NMF). Import both beta-divergence implementations from the `torchbd` package as follows:

```python
# Import PyTorch beta-divergence implementations
from torchbd.loss import *

```


### Beta-divergence between two matrices

To calculate the beta-divergence between matrix `A` and a target or reference matrix `B`, use the `BetaDivLoss` loss function. The `BetaDivLoss` loss function can be instantiated and used as follows:

```python
# Instantiate beta-divergence loss object
beta_div_loss = BetaDivLoss(beta=0, reduction='mean')

# Calculate beta-divergence loss between matrix A and target matrix B
loss = beta_div_loss(input=A, target=B)

```


### NMF beta-divergence between data matrix and reconstruction

To calculate the NMF-specific beta-divergence between data matrix `X` and the matrix product of a scores matrix `H` and a components matrix `W`, use the `NMFBetaDivLoss` loss function. The `NMFBetaDivLoss` loss function can be instantiated and used as follows:

```python
# Instantiate NMF beta-divergence loss object
nmf_beta_div_loss = NMFBetaDivLoss(beta=0, reduction='mean')

# Calculate beta-divergence loss between data matrix X (target or
# reference matrix) and matrix product of H and W
loss = nmf_beta_div_loss(X=X, H=H, W=W)

```


### Choosing beta value

When instantiating beta-divergence loss objects, the value of beta should be chosen depending on data type and application. For NMF applications, a beta value of 0 (Itakura-Saito divergence) is recommemded. Integer values of beta correspond to the following divergences and loss functions:

* beta = 0: [Itakura-Saito divergence](https://en.wikipedia.org/wiki/Itakura-Saito_distance)
* beta = 1: [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback-Leibler_divergence)
* beta = 2: [mean-squared error](https://en.wikipedia.org/wiki/Mean_squared_error)


## Issue Tracking and Reports

Please use the [GitHub issue tracker](https://github.com/wecarsoniv/pytorch-beta-divergence/issues) associated with this repository for issue tracking, filing bug reports, and asking general questions about the package or project.

