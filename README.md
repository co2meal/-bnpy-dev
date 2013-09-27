bnpy is Bayesian nonparametric unsupervised machine learning for python.

Contact:  Mike Hughes. mike AT michaelchughes.com 

# About
This python module provides code for training popular Bayesian nonparametric models on massive datasets. bnpy supports the latest online learning algorithms as well as standard offline methods. 

Support probabilistic models include

* Gaussian mixture models
    * parametric and nonparametric (Dirichlet Process)

Supported learning algorithms include:

* EM: full-dataset (batch) EM
* VB: full-dataset (batch) VB
* moVB: memoized online VB
* soVB: stochastic online VB

These are all variants of *variational inference*, a family of optimization algorithms that perform coordinate ascent to learn parameters. 

# Quick Start

Consider our simple "AsteriskK8" toy dataset.  

To compare an 8-component standard Gaussian mixture model (GMM) trained with EM to a Dirichlet process GMM trained with memoized online VB, we'd run

```
python Learn.py AsteriskK8 MixModel Gauss EM --K 5
python Learn.py AsteriskK8 DPMixModel Gauss moVB --K 5
```

# Installation

Follow the [installation instructions](https://bitbucket.org/michaelchughes/bnpy/wiki/Installation.md) on our project wiki.

# Documentation

All documentation can be found on the  [project wiki](https://bitbucket.org/michaelchughes/bnpy/wiki/Home.md).

Especially check out the [quick start demos](https://bitbucket.org/michaelchughes/bnpy/wiki/QuickStart/QuickStart.md)

# Target Audience

Primarly, we intend bnpy to be a platform for researchers pushing the state-of-the-art. By gathering many learning algorithms and popular models in one convenient, flexible repository, we hope to make it easier to cmopare and contrast approaches.

# Repository Organization
  bnpy/ module-specific code

  config/ configuration files, specifying default options for all models and learning algorithms

  profile/ utilities for profiling run-time performance and identifying bottlenecks

  tests/ unit-tests for assuring code correctness. using nose package.

