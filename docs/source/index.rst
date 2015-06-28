.. bnpy documentation master file, created by
   sphinx-quickstart on Mon Mar  2 15:33:33 2015.
   Must contain the root `toctree` directive.

bnpy : Bayesian nonparametric clustering for Python
===============

Our goal is to make it easy for 
Python programmers 
to train state-of-the-art clustering models on large datasets.
We focus on nonparametric models based on the Dirichlet process, especially extensions that handle hierarchical and sequential datasets.
Traditional parametric counterparts (like finite mixture models) are also supported. 


Training a model with **bnpy** requires the user to specify the dataset, the model, and the algorithm to use. Flexible keyword options allow advanced users lots of control, but smart defaults make it simple for beginners. 
**bnpy**'s modular implementation makes it possible to try many variants of models and algorithms, to find the best fit for the data at hand.

For example, to train a 25 component Gaussian mixture model via EM on your dataset:

.. code-block:: python

  >>> python -m bnpy.Run MyDataset.py FiniteMixtureModel Gauss EM --K 25

Alternatively, we can try a **Dirichlet process**-based Gaussian mixture model:

.. code-block:: bash

	$ python -m bnpy.Run MyDataset.py DPMixtureModel Gauss VB --K 25

Next, we could take advantage of sequential structure in your data by training a hidden Markov model using memoized variational inference.

.. code-block:: bash

	$ python -m bnpy.Run MyDataset.py HDPHMM DiagGauss moVB --moves merge,delete

.. code-block:: python

  >>> bnpy.run(MyDatasetObject, 'HDPHMM', 'DiagGauss', 'moVB', moves='merge,delete')


Supported clustering models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Mixture models
    * `FiniteMixtureModel` : fixed number of clusters
    * `DPMixtureModel` : infinite number of clusters, via the Dirichlet process

* Topic models (aka admixtures models)
    * `FiniteTopicModel` : fixed number of topics. This is Latent Dirichlet allocation.
    * `HDPTopicModel` : infinite number of topics, via the hierarchical Dirichlet process
    
* Hidden Markov models (HMMs)
    * `FiniteHMM` : Markov sequence model with a fixture number of states
    *  `HDPHMM` : Markov sequence models with an infinite number of states

* COMING SOON
    * relational models (like the IRM, MMSB, etc.)
    * grammar models


Supported data-generation models (likelihoods)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Real-valued vector observations (1-dim, 2-dim, ... D-dim)
    * `Gauss` : Full-covariance Gaussian
    * `DiagGauss` : Diagonal-covariance Gaussian
    * `ZeroMeanGauss` : Zero-mean, full-covariance
    * `AutoRegGauss` : first-order auto-regressive Gaussian 
* Binary vector observations (1-dim, 2-dim, ... D-dim)
    * `Bern` : Bernoulli 
* Discrete, bag-of-words data (each observation is one of V symbols)
    * `Mult` : Multinomial


Supported algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Variational methods
    * `EM` : Expectation-maximization
    * `VB` : variational Bayes
    * `soVB` : stochastic variational (online)
    * `moVB` : memoized variational (online)

* COMING SOON
    * Gibbs sampling

Why bnpy?
======================================
TODO



.. toctree::
    :maxdepth: 2

    Installation
    UserGuide
    demos/DemoIndex
    

