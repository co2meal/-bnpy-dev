===========================================
DP Mixtures: VB Learning Technical Details 
===========================================

Here, we review the probabilistic generative model of the DP mixtures,
and present complete equations for standard mean-field variational learning,
as presented in [cite].

Model Hyperparameters
~~~~~~~~~~~~~~~~~~~~~

* Concentration parameter :math:`{\gamma}`
	Positive scalar	concentration parameter of Dirichlet process.

This value controls the expected number of states assigned to data items
under the prior. Typically, a reasonable value would be larger than 1
but less than 100.

In code, the ``DPMixtureModel`` class has a scalar attribute ``gamma0``.

You can specify this as a keyword arg: ``--gamma0``.


Model Random Variables
~~~~~~~~~~~~~~~~~~~~~~

* State conditional probabilities :math:`\{u_k\}_{k=1}^{\infty}`
	Positive numbers between 0 and 1, with one scalar for each component.

We can interpret each value :math:`u_k` as a conditional probability of
choosing state k among the infinite set of options [k, k+1, k+2, ...].

The generative model for this variable is simply

.. math::
	u_k \sim \mbox{Beta}(1, \gamma)


* State probabilities :math:`{\beta}_{k=1}^{\infty}`
	Positive numbers that sum to one, one scalar for each component.

This collection has an entry for each of the countably infinite components.
However, we can write this collection as a finite vector :math:`{\beta}` 
of size K+1, by defining the last index to aggregate 
all entries larger than index K.

.. math::
	\beta = [ \beta_1, \beta_2, \ldots \beta_K \beta_{>K} ]
	\\ \beta_1 \triangleq p(z_n = 1)
	\\ \ldots
	\\ \beta_K \triangleq p(z_n = K)
	\\ \beta_{>K} \triangleq \sum_{\ell=K+1}^{\infty} p(z_n = \ell)

This variable is determined completely by the value of :math:`u`, via the stick-breaking construction:

For each component k = 1, 2, 3, ...

.. math::
	\beta_k = u_k \prod_{\ell=1}{\ell<k} (1 - u_\ell)

* Data-cluster assignments :math:`{z}`
	Integer :math:`z_n` indicates the 
	component to which data item n is assigned.

.. math::
	z = [ z_1, z_2, \ldots z_N ], \quad	z_n \in \{1, 2, 3, 4, \ldots\}

The generative model for this variable is

.. math::
	z_n \sim \mbox{Discrete}(\beta_1, \beta_2, \ldots \beta_K \ldots)

Complete Joint Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is the complete joint density of assignments :math:`z` and 
conditional probabilities :math:`u`:

.. math ::
	p(z, u) = 
	\prod_{k=1}^K \mbox{Beta}(u_k | 1, \gamma)
	\prod_{n=1}^N \mbox{Discrete}(z_n | \beta(u) )

Remember that component probabilities :math:`{\beta}` 
are functionally determined by :math:`u`, via Eq. TODO.

Variational Parameters
~~~~~~~~~~~~~~~~~~~~~~

* Local assignments :math:`r` for factor :math:`q(z)`

.. math ::
	q(z) = \prod_{n=1}^N 
	\mbox{Discrete}(z_n | r_{n1}, r_{n2}, \ldots r_{nK})

Each data item has a vector :math:`r_n` of size K. 
Each entry is positive, and the sum of this vector is one.

We can interpret :math:`r_{nk}` as the posterior probability that 
data item n is explained by cluster k.

In **bnpy** code, ``resp`` is a 2D array with size N x K.

* Global parameters :math:`\eta` for factor :math:`q(u)`

.. math ::
	q(u) = \prod_{k=1}^K \mbox{Beta}(u_k | \eta_{k1}, \eta_{k0})

Each component k has two positive scalars, 
:math:`\eta_{k1}` and :math:`\eta_{k0}`.

We can interpret :math:`\eta_{k1}` as a pseudo-count of the number
of times we have seen assignments to state k across the corpus. 
Similarly, :math:`\eta_{k0}` can be interpreted as the number of
times we have seen assignments to a state with index larger than k.

In code, ``eta1`` is a 1D array of size K, and 
``eta0`` is a 1D array of size K.

Sufficient statistics
~~~~~~~~~~~~~~~~~~

As usual, for this mixture model we can summarize all information about local 
assignments needed for global updates as vectors of counts. We define the following for each component k = 1, 2, ... K:

* Count :math:`N_k`
	expected number of assignments to component $k$

* Count :math:`N_{>k}`
	expected number of assignments to component with index larger
	than k

Formally, we have

.. math::
	N_k &\triangleq \sum_{n=1}^N r_{nk}
	\\
	N_{>k} &\triangleq \sum_{n=1}^N \sum_{\ell = k+1}^{K} r_{n\ell}

Under our truncation assumption, :math:`N_{>K}` equals 0.

In code, we represent :math:`N = [N_1, N_2, \ldots N_K]` via a 1D array 
stored as a field in sufficient statistics: ``SS.N``. Given ``SS.N``,
we can compute :math:`N_{>}` via the function ``N_gt()``.


Useful expectations
~~~~~~~~~~~~~~~~~~~

* Expected log conditional probabilities for state k: :math:`E_q[ \log u_k]`
	This is a function of :math:`\eta_{k1}, \eta_{k0}`.

.. math::
	E_q[ \log u_k ] &= \psi( \eta_{k1} ) - \psi(\eta_{k1}+\eta_{k0})
	\\
	E_q[ \log 1 - u_k ] &= \psi( \eta_{k0} ) - \psi(\eta_{k1}+\eta_{k0})


* Expected log probability of state k: :math:`E[ \log \beta_k]`
	Under :math:`q(u)`, this is a function of the entire vectors :math:`\eta`.

For each component k = 1, 2, ... K:

.. math::
	E_q[ \log \beta_k ] &= E_q[ \log u_k ] + \sum_{\ell=1}^{k-1} \E_q[ \log 1 - u_\ell]
	:label:Elogbeta

where we can substitute definitions above. 


Objective Function for VB
~~~~~~~~~~~~~~~~~~~~~~~~~

Our goal in variational learning is to find local assignments :math:`r`
and global parameters :math:`{\eta}` that maximize our ELBO objective
function. This function decomposes into entropy and allocation terms.

* The allocation term :math:`\mathcal{L}_{alloc}`
	collects all terms that are linear functions of sufficient statistics
	and global parameters :math:`\eta`.

.. math::
	\mathcal{L}(r, \eta) 
	&\triangleq 
	E_q 
	\Big[ \log p(z) + 
	\sum_{k=1}^K 
	\log \frac{p(u_k | 1, \gamma)}{q(u_k | \eta_{k1}, \eta_{k0})} \Big]
	\\
	&= \sum_{k=1}^K \Big( c_B(1, \gamma) - c_B(\eta_{k1}, \eta_{k0})
	\\
	&\qquad + (N_{k}(r) + 1 - \eta_{k1}) E_q[ \log u_k ]
	+ (N_{>k}(r) + \gamma_0 - \eta_{k0}) E_q[ \log 1-u_k ]
	\Big)

The function \\(c_B\\) is the cumulant function of the Beta distribution, 
which takes as arguments two positive scalars \\(a, b \\).

.. math::
	c_B(a, b) &\triangleq 
	\log \Gamma(a + b)
	 - \log \Gamma(a)
	 - \log \Gamma(b)

The required expectations are functions of :math:`{\eta}` defined above in :eq: Elogu. 

* The entropy term :math:`\mathcal{L}_{entropy}`
	collects all terms that must be computed directly from local parameters,
	without the sufficient statistics already used for global updates.

.. math::
	\mathcal{L}_{entropy}(r)
	&\triangleq
	- \sum_{n=1}^N E_q[ \log q(z_n) ]
	\\
	&= - \sum_{n=1}^N \sum_{k=1}^K r_{nk} \log r_{nk}

This whole function produces a scalar that indicates the entropy (level of uncertainty) in the provided assignments :math:`r`. This scalar will always be non-negative, with value increasing as assignments are more uncertain.

In practice, we can simplify this computation across each component:

.. math::
	\mathcal{L}_{entropy}(r) &= \sum_{k=1}^K H_k
	\\
	H_k \triangleq \sum_{n=1} r_{nk} \log r_{nk}

Where each value :math:`H_k` satisfies :math:`H_k > 0`.
	

Global Step Update
~~~~~~~~~~~~~~~~~~

Following standard techniques, we update the global parameters :math:`{\eta}` for each component k = 1, 2, ... K:

.. math::
	\eta_{k1} &= N_{k}(r) + 1
	\\
	\eta_{k0} &= N_{>k}(r) + \gamma_0

Local Step Update
~~~~~~~~~~~~~~~~~~

Updates to assignment vector :math:`r_{n}` require as input the current
fixed global parameters :math:`\eta` and the log soft evidence 
vector :math:`L_n` computed by the observation model.

For each component k = 1, 2, ... K:

.. math::
	\tilde{r}_{nk} &= \exp \Big( E_{q}[ \log \beta_k] + L_{nk} \Big)
	\\
	r_{nk} &= \frac{\tilde{r}_{nk}}{\sum_{\ell=1}^K \tilde{r}_{n\ell}}

See :eq: Elogbeta for the definition of the required expectation.

