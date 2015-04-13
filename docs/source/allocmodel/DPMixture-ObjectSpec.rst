===========================================
DP Mixtures: Practical Introduction
===========================================

The DPMixtureModel Object
-------------------------

Attributes for Prior Hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``gamma0`` : float

Attributes for VB learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``K`` : int
	number of active components to instantiate.

* ``eta1`` : 1D array, size K
	Parameter of approximate posterior factor q(u_k).
	q(u_k) = Beta(u_k | eta1[k], eta0[k])

* ``eta0`` : 1D array, size K
	Parameter of approximate posterior factor q(u_k).
	q(u_k) = Beta(u_k | eta1[k], eta0[k])

Fields in local parameter (LP) dict
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``resp`` : 2D array, size N x K
	Parameters of approximate posterior factor q(z_n) for n = 0 ... N-1.

	q(z_n) = Discrete(z_n | resp[n,0], ... resp[n,K-1])

	Entry n,k gives posterior probability that component n is assigned
	to component k

Methods for VB learning:
~~~~~~~~~~~~~~~~~~~~~~~~

* ``calc_local_params``
	Update ``resp`` field to best value under objective function,
	given current global parameters.

* ``update_global_params``
	Update ``eta1, eta0`` to best values under objective function,
	given provided sufficient statistics.


Methods for initialization:
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Initialization sets the attribute for VB learning to valid values.

* ``init_global_params``	
	Initialize ``eta1, eta0`` so that expected probability
	of each component is roughly uniform.
* ``set_global_params``
	Initialize from provided values of `eta1,eta0`.


Options for Initialization
~~~~~~~~~~~~~~~~~~~~~~

Under any "from scratch" initialization, like randexamples or randexamplesbydist, the method `init_global_params` is called.
This method initializes the global parameters so the expected value of probabilities :math:`{\beta}` is nearly uniform:

.. math::
	\beta_k \approx \frac{1}{K}

Alternatively, you can always initialize values via either `set_global_params` or `update_global_params`.



