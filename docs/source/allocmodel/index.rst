=========================
Allocation Model Overview
=========================

What is an allocation model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These components are responsible for:

* Objective function
	computing all terms that do not belong in the observation model. 

* Local step
	updating local variational parameters for a chunk of data. 

* Summary step
  computing sufficient statistics from local parameters.

* Global step
  updating global parameters given sufficient statistics.

.. toctree::
   :maxdepth: 3
   :titlesonly:

   MixtureModels-Overview
   TopicModels-Overview
   HiddenMarkovModels-Overview
