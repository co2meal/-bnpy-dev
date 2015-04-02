.. bnpy documentation master file, created by
   sphinx-quickstart on Mon Mar  2 15:33:33 2015.
   Must contain the root `toctree` directive.

Welcome to bnpy
===============

Quick Links
~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   Installation
   demos/DemoIndex
   allocmodel/index
   obsmodel/index

Purpose: Hierarchical Dirichlet Process clustering made easy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our goal is to make it easy to train popular clustering models on large datasets, so you can try many different models and find one that works best. 
We focus on nonparametric models based on the Dirichlet process
and extensions that handle hierarchical and sequential datasets.
We also provide traditional parametric (finite) counterparts.

Training a model with **bnpy** is as simple as specifying the dataset (as a python script), names for the desired model components, and the algorithm to use.  

For example, to train a Gaussian mixture model via EM on your dataset:

.. code-block:: bash

	$ python -m bnpy.Run MyDataset.py FiniteMixtureModel Gauss EM

Alternatively, you can easily use the Dirichlet Process mixture model and the 
variational Bayes (VB) algorithm:

.. code-block:: bash

	$ python -m bnpy.Run MyDataset.py DPMixtureModel Gauss VB

You could even take advantage of sequential structure in your data by training a Markov model using memoized variational inference, with delete and merge moves:

.. code-block:: bash

	$ python -m bnpy.Run MyDataset.py HDPHMM DiagGauss moVB --moves merge,delete


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

* Bernoulli for binary data
    * `Bern`
* Multinomial for discrete, bag-of-words data
    * `Mult`
* Gaussian for real-valued vector data
    * `Gauss` : Full-covariance 
    * `DiagGauss` : Diagonal-covariance
    * `ZeroMeanGauss` : Zero-mean, full-covariance
* Auto-regressive Gaussian
    * `AutoRegGauss`


Goals of this page:
* High level overview of bnpy capabilities
* Why bnpy and not some other package?
* Show off demo gallery (like Bokeh)?


Index
==================

* :ref:`genindex`

