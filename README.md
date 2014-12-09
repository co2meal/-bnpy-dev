**bnpy** is Bayesian nonparametric unsupervised machine learning for python.

Contact:  Mike Hughes. Email: mike AT michaelchughes.com 

# About
This python module provides code for training popular Bayesian nonparametric models on massive datasets. **bnpy** supports the latest online learning algorithms as well as standard offline methods. 

### Supported probabilistic models

* Mixture models
    * `FiniteMixtureModel` : fixed number of clusters
    * `DPMixtureModel` : infinite number of clusters, via the Dirichlet process

* Topic models (aka admixtures models)
    * `FiniteTopicModel` : fixed number of topics. This is Latent Dirichlet allocation.
    * `HDPTopicModel` : infinite number of topics, via the hierarchical Dirichlet process
    
* Hidden Markov models (HMMs)
    * `FiniteHMM` : fixture number of states
    * COMING SOON `HDPHMM` : infinite number of states

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

# Quick Start

**bnpy** provides an easy command-line interface for launching experiments.

Train 8-component Gaussian mixture model via the offline EM algorithm.

```
python -m bnpy.Run AsteriskK8 FiniteMixtureModel ZMGauss EM --K 8
```

Train Dirichlet-process Gaussian mixture model (DP-GMM) via full-dataset variational algorithm.

```
python -m bnpy.Run AsteriskK8 DPMixtureModel Gauss VB --K 8
```

Train DP-GMM via memoized variational, with birth and merge moves.

```
python -m bnpy.Run AsteriskK8 DPMixtureModel Gauss moVB --moves birth,merge
```

### Quick help
```
# print help message for required arguments
python -m bnpy.Run --help 
# print help message for specific keyword options for Gaussian mixture models
python -m bnpy.Run AsteriskK8 MixModel Gauss EM --kwhelp
```

# Installation

Follow the [installation instructions](https://bitbucket.org/michaelchughes/bnpy/wiki/Installation.md) on our project wiki.

# Documentation

All documentation can be found on the  [project wiki](https://bitbucket.org/michaelchughes/bnpy/wiki/Home.md).

Especially check out the [quick start demos](https://bitbucket.org/michaelchughes/bnpy/wiki/QuickStart/QuickStart.md)

# Target Audience

Primarly, we intend **bnpy** to be a platform for researchers. By gathering many learning algorithms and popular models in one convenient, modular repository, we hope to make it easier to compare and contrast approaches.

# Repository Organization

* bnpy/ : module-specific code

* datasets/ : example datasets and scripts for generating toy data

* tests/ : unit-tests for assuring code correctness. using nose package.

