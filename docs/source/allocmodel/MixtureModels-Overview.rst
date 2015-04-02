=======================
Mixture Models Overview
=======================

Supported Data Formats
~~~~~~~~~~~~~~~~~~~~~~~

Mixture models can apply to almost all data formats available in bnpy.
Any data suitable for topic models or sequence models can also be fit
with a basic mixture model.

The only formats that do not apply are those based on GraphData, 
which require the subclass of mixture models (TBD).

Supported Learning Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Currently, the practical differences are:

* `FiniteMixtureModel` supports EM, VB, soVB, moVB
* `DPMixtureModel` supports VB, soVB, and moVB.
* * with birth/merge/delete moves for moVB

EM (MAP) inference for the DPMixtureModel is possible, but just not implemented yet.

Model Comparison
~~~~~~~~~~~~~~~~~~~~~~~

There are two types of mixture model supported. Both define the model in 
terms of a global parameter vector :math:`\beta`, where :math:`\beta_k` gives the probability of topic k, and local assignments :math:`z`, where :math:`z_n` indicates which state {1, 2, 3, ... K} is assigned to data item n.

The `FiniteMixtureModel` has a generative process:

.. math::
	[\beta_1, \beta_2, \ldots \beta_K] 
	\sim \mbox{Dir}(\gamma, \gamma, \ldots \gamma)
	\\
	z_n \sim \mbox{Discrete}(\beta)

while the `DPMixtureModel` has generative process:

.. math::
	[\beta_1, \beta_2, \ldots \beta_K \ldots] 
	\sim \mbox{StickBreaking}(\gamma_0)
	\\
	z_n \sim \mbox{Discrete}(\beta)

If we let K grow to infinity, these two models converge if :math:`\gamma = \gamma_0 /K`.


Sufficient Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~

Mixture models produce the following sufficient statistics:

* Count N_k
	Expected assignments to state k across all data items.


Mixture Model Contents
~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   FiniteMixture-Practical.rst
   FiniteMixture-EMDetails.rst
   FiniteMixture-VBDetails.rst

   DPMixture-Practical.rst
   DPMixture-VBDetails.rst