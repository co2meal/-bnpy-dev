===========================================
DPMixtureModel: Practical Introduction
===========================================

The :class:`.DPMixtureModel` class represents the allocation model for Dirichlet Process mixtures. For the generative process, 
we assume a stick-breaking construction for global probability vector :math:`\pi_0`, 
and then each data atom draws its cluster assignment :math:`z_n` i.i.d. given these probabilities.
While the true generative process has an unbounded number of clusters, in practice, we truncate to :math:`K` active clusters. The value of K is specified at initialization by the end-user, and cannot change once fixed in standard optimization methods. Our new adaptive proposal moves can change the value of K as the algorithm sees more data.


The DPMixtureModel Object
-------------------------

Attributes : Prior Hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``gamma`` : float
	Concentration parameter :math:`\gamma > 0` 

Attributes : Variational Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``K`` : int
	number of active clusters

* ``eta1`` : 1D array, size K
	Global variational parameter.
	"On" pseudo-count of Beta posterior on stick-breaking weights for each cluster.

* ``eta0`` : 1D array, size K
	Global variational parameter.
	"Off" pseudo-count of Beta posterior on stick-breaking weights for each cluster.

.. math::

	for k in 1, 2, \ldots K:

		q(u_k) = Beta(u_k | \eta_{1k}, \eta_{0k})

		E[u_k] = \eta_{1k} / (\eta_{1k} + \eta_{0k})

		E[\pi_{0k}] = E[u_k] \prod_{\ell=1}^{\ell = k - 1} (1 - E[u_{\ell}])


Local Parameters (LP)
~~~~~~~~~~~~~~~~~~~~~

The following represent named fields inside the local parameter dict ``LP``:

* ``resp`` : 2D array, size N x K
	Local assignment parameters.

	Each row ``resp[n,:]`` provides parameters for the approximate posterior  for discrete assignment indicator :math:`z_n`.

	.. math::
		q(z_n) = \mbox{Cat}(z_n | r_{n1}, r_{n2}, \ldots r_{nK} )

	Entry ``resp[n,k]`` can be interpreted the posterior probability that cluster k is assigned to data atom n. We sometimes call this the *responsibility* cluster k bears for data atom n.

	To be a valid parameter for this Categorical distribution, entries in ``resp`` must be non-negative, and each row must sum to one.
	
Summaries (SS) for global updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following represent summary fields used for updating global parameters. 
These are computed by ``get_global_suff_stats`` and returned as an instance of  :class:`.SuffStatBag` ``SS``.  

You can access each field via dot notation. For example: ``SS.N``.

* ``N`` : 1D array, size K
	Expected posterior number of data atoms assigned to each cluster.

	.. math::
		N_k = \sum_{n=1}^N r_{nk}

Summaries (SS) for ELBO Terms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following represent summary fields used for computing the variational objective function.
These are computed by ``get_global_suff_stats`` when called with keyword and returned as an instance of  :class:`.SuffStatBag` ``SS``.  

You can access ELBO fields by calling the :any:`SuffStatBag.getELBOTerm` method. For example: ``SS.getELBOTerm("Hresp")``.

* ``Hresp`` : 1D array, size K
	Entropy value for assignments to cluster k.

	.. math::
		H_k = - \sum_{n=1}^N r_{nk} \log r_{nk}


Methods : Variational learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Local Step
++++++++++

* :any:`DPMixtureModel.calc_local_params`


Summary Step
++++++++++++

Global Step
+++++++++++

* ``update_global_params``
	Update ``eta1, eta0`` to best values under objective function,
	given provided sufficient statistics.

Objective Step
++++++++++++++



Methods : initialization
~~~~~~~~~~~~~~~~~~~~~~~~

When a DPMixtureModel object is created, it has only attributes related to prior hyperparameters. None of the attributes needed for a learning algorithm, including K, exist yet. The process of *initialization* fills in the relevant attributes.

There are three methods to properly initialize a DPMixtureModel:

* :any:`DPMixtureModel.init_global_params`	
	Initialize global parameters so that expected probability vector :math:`\pi_0` is roughly uniform.

* :any:`DPMixtureModel.set_global_params`	
	Initialize global parameters to user-specified values via keyword arguments.

* :any:`DPMixtureModel.update_global_params`	
	Create global parameters that optimize the learning objective,
	given user-specified summaries SS.



Keywords Args for Training
--------------------------

Each named option below can be provided to ::class::`bnpy.Run` as a keyword, value pair to influence training of a ``DPMixtureModel``.

Hyperparameters
~~~~~~~~~~~~~~~

* ``gamma`` : float
	Concentration parameter. Larger values tend to learn more clusters.
	Default is 1.0.

Initialization
~~~~~~~~~~~~~~

* ``K`` : int
	Number of active clusters.


