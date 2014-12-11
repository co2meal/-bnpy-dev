**bnpy** is Bayesian nonparametric unsupervised machine learning for python.

# About
This python module provides code for training popular clustering models on large datasets. We focus on Bayesian nonparametric models based on the Dirichlet process, but also provide parametric counterparts as well.

**bnpy** supports the latest online learning algorithms as well as standard offline methods. 

### Supported probabilistic models

* Mixture models
    * `FiniteMixtureModel` : fixed number of clusters
    * `DPMixtureModel` : infinite number of clusters, via the Dirichlet process

* Topic models (aka admixtures models)
    * `FiniteTopicModel` : fixed number of topics. This is Latent Dirichlet allocation.
    * `HDPTopicModel` : infinite number of topics, via the hierarchical Dirichlet process
    
* Hidden Markov models (HMMs)
    * `FiniteHMM` : Markov sequence model with a fixture number of states

* **COMING SOON**
    *  `HDPHMM` : Markov sequence models with an infinite number of states
    * grammar models
    * relational models


### Supported data-generating models (aka likelihoods)

* Multinomial for bag-of-words data
    * `Mult`
* Gaussian for real-valued vector data
    * `Gauss` : Full-covariance 
    * `DiagGauss` : Diagonal-covariance
    * `ZeroMeanGauss` : Zero-mean, full-covariance
* Auto-regressive Gaussian
    * `AutoRegGauss`

### Supported learning algorithms:

* Expectation-maximization (offline)
    * `EM`
* Full-dataset variational Bayes (offline)
    * `VB`
* Memoized variational (online)
    * `moVB`
* Stochastic variational (online)
    * `soVB`

These are all variants of *variational inference*, a family of optimization algorithms. We plan to eventually support sampling methods (Markov chain Monte Carlo) too.

# Demos

You can find lots of examples of **bnpy** in action in our curated set of  [IPython notebooks]
(http://nbviewer.ipython.org/urls/bitbucket.org/michaelchughes/bnpy-dev/raw/master/demos/DemoIndex.ipynb).

These same demos are also directly available on our [wiki] (https://bitbucket.org/michaelchughes/bnpy/wiki/demos/DemoIndex.rst).

# Quick Start

You can use **bnpy** from the terminal, or from within Python. Both options require specifying a dataset, an allocation model, an observation model (likelihood), and an algorithm. Optional keyword arguments with reasonable defaults allow control of specific model hyperparameters, algorithm parameters, etc.

Below, we show how to call bnpy to train a 8 component Gaussian mixture model on the default AsteriskK8 toy dataset (shown below).
In both cases, log information is printed to stdout, and all learned model parameters are saved to disk.

## Calling from the terminal/command-line

```
$ python -m bnpy.Run AsteriskK8 FiniteMixtureModel Gauss EM --K 8
```

## Calling directly from Python

```
import bnpy
bnpy.run('AsteriskK8', 'FiniteMixtureModel', 'Gauss', 'EM', K=8)
```

## Other examples
Train Dirichlet-process Gaussian mixture model (DP-GMM) via full-dataset variational algorithm (aka "VB" for variational Bayes).

```
python -m bnpy.Run AsteriskK8 DPMixtureModel Gauss VB --K 8
```

Train DP-GMM via memoized variational, with birth and merge moves, with data divided into 10 batches.

```
python -m bnpy.Run AsteriskK8 DPMixtureModel Gauss moVB --K 8 --nBatch 10 --moves birth,merge
```

## Quick help
```
# print help message for required arguments
python -m bnpy.Run --help 

# print help message for specific keyword options for Gaussian mixture models
python -m bnpy.Run AsteriskK8 MixModel Gauss EM --kwhelp
```

# Installation

To use **bnpy** for the first time, follow the [installation instructions](https://bitbucket.org/michaelchughes/bnpy/wiki/Installation.md) on our project wiki.

Once installed, please visit the [Configuration](https://bitbucket.org/michaelchughes/bnpy/wiki/Configuration.md) wiki page to learn how to configure where data is saved and loaded from on disk.

# Documentation

All documentation can be found on the  [project wiki](https://bitbucket.org/michaelchughes/bnpy/wiki/Home.md).

# Contributors

Primary contact
* Mike Hughes (mike@michaelchughes.com)

Contributors at Brown CS
* William Stephenson
* Sonia Phene
* Mert Tertizhan
* Mengrui Ni
* Geng Ji
* Jincheng Li


# Target Audience

Primarly, we intend **bnpy** to be a platform for researchers. 
By gathering many learning algorithms and popular models in one convenient, modular repository, we hope to make it easier to compare and contrast approaches.
We also how that the modular organization of **bnpy** enables researchers to try out new modeling ideas without reinventing the wheel.
