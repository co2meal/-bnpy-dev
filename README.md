bnpy is Bayesian nonparametric unsupervised machine learning for python
Author: Mike Hughes, mike@michaelchughes.com 

# About
This python module provides code for training popular hierarchical Bayesian models from large datasets. We especially emphasize non-parametric models, such as the DP or HDP.

Current the following models are supported:
** Gaussian mixture models (parametric and Dirichlet-process)

Learning is done via variational inference, an optimization-based method. The supported learning algorithms include
** moVB: memoized online VB
** soVB: stochastic online VB
** VB: full-dataset (batch) VB
** EM: full-dataset (batch) EM

# Quick Start
From the command line, 
>> python Learn.py <my dataset> MixModel Gaussian VB <<options>>

# Repository Organization
  bnpy/ module-specific code

  config/ configuration files, specifying default options for all models and learning algorithms

  profile/ utilities for profiling run-time performance and identifying bottlenecks

  tests/ unit-tests for assuring code correctness. using nose package.

