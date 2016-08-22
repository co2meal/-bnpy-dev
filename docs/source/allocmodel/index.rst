=================
Allocation Models
=================

What is an allocation model?
----------------------------

An allocation model describes how discrete clusters are assigned to a dataset, even if the data has hierarchical/sequential structure.
For a basic introduction to these concepts, see the Compositional Models documentation.

The core of an allocation model is a *generative* probabilistic model, which produces cluster assignments :math:`z_1, z_2, \ldots z_N` for each observation in the dataset.

.. math::
    p( z | \pi ) = ???

Each allocation model provides the following essential functionality:

* Objective function
    The function optimized by the training algorithm given observed data.

* Local step
	Updating local parameters for a chunk of data. 

* Summary step
    computing sufficient statistics from local parameters.

* Global step
    updating global parameters given sufficient statistics.


Supported allocation models
---------------------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   MixtureModels-Overview
   TopicModels-Overview
