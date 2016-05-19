import numpy as np


class SupervisedFiniteTopicModel(AllocModel):

    '''
    Bayesian nonparametric topic model with a K active components.

    Uses a direct construction that truncates unbounded posterior to
    K active components (assigned to data), indexed 0, 1, ... K-1.
    Remaining mass for inactive topics is represented at index K.

    Attributes
    -------
    inferType : string 'VB' (TODO: include {'VB', 'moVB', 'soVB'})
        indicates which updates to perform for local/global steps
    K : int
        number of components
    alpha : float
        scalar pseudo-count
        used in Dirichlet prior on document-topic probabilities.
    delta : float
        scalar concentration parameter.

    Attributes for VB
    ---------
   

    Variational Local Parameters
    --------
    resp :  2D array, N x K
        q(z_n) = Categorical( resp_{n1}, ... resp_{nK} )
    theta : 2D array, nDoc x K
        q(pi_d) = Dirichlet( \theta_{d1}, ... \theta_{dK} )

    References
    -------
    Latent Dirichlet Allocation, by Blei, Ng, and Jordan
    introduces a classic topic model with Dirichlet-Mult observations.
    Supervised Latent Dirichlet Allocation by Blei and McAuliffe.
    
    '''

    def __init__(self, inferType, priorDict=None):
        if inferType == 'EM':
            raise ValueError('SupervisedFiniteTopicModel cannot do EM.')
        if inferType != 'VB':
			 raise ValueError('SupervisedFiniteTopicModel can only do VB.')
        self.inferType = inferType
        self.K = 0
        if priorDict is None:
            self.set_prior()
        else:
            self.set_prior(**priorDict)


    def get_active_comp_probs(self):
        ''' Get vector of appearance probabilities for each active comp.

        Returns
        -------
        beta : 1D array, size K
            beta[k] gives probability of comp. k under this model.
        '''
        return np.ones(self.K) / float(self.K)
        

    def set_prior(self, alpha=1.0, eta=None, delta=1.0, **kwargs):
        self.alpha = float(alpha)
        
        self.delta = float(delta)
        if eta is not None:
        	self.eta = float(eta)
        else:
        	self.eta = np.ones(self.K)

    def to_dict(self):
        return dict()

    def from_dict(self, Dict):
        self.inferType = Dict['inferType']
        self.K = Dict['K']
        self.eta = Dict['eta']
        self.delta = Dict['delta']
        
    def get_prior_dict(self):
        return dict(alpha=self.alpha,
                    K=self.K,
                    inferType=self.inferType,
                    eta=self.eta,
                    delta=self.delta)

    def get_info_string(self):
		''' Returns human-readable name of this object
		'''
		return 'Finite Supervised LDA model with K=%d comps. alpha=%.2f, delta=%.2f' \
		    % (self.K, self.alpha, self.delta)

    
    def calc_local_params(self, Data, LP, **kwargs):
        ''' Calculate document-specific quantities (E-step)

        Parameters
        -------
        Data : bnpy.data.DataObj subclass
        LP : dict
            Local parameters as key-value string/array pairs
            * E_log_soft_ev : 2D array, N x K
                E_log_soft_ev[n,k] = log p(data obs n | comp k)

        Returns
        -------
        LP : dict
            Local parameters, with updated fields
            * resp : 2D array, N x K
                Posterior responsibility each comp has for each item
                resp[n, k] = p(z[n] = k | x[n])
            * theta : 2D array, nDoc x K
                Positive pseudo-count parameter for active topics,
                in the approximate posterior on doc-topic probabilities.
            * thetaRem : scalar float
                Positive pseudo-count parameter for inactive topics.
                in the approximate posterior on doc-topic probabilities.
            * ElogPi : 2D array, nDoc x K
                Expected value E[log pi[d,k]] under q(pi).
                This is a function of theta and thetaRem.
        '''
        raise Exception('calc_local_params is not implemented.')
        '''
        alphaEbeta = self.alpha_E_beta()
        alphaEbetaRem = self.alpha_E_beta_rem()
        assert np.allclose(alphaEbeta[0], 
            self.alpha * self.rho[0])
        if alphaEbeta.size > 1:
            assert np.allclose(alphaEbeta[1], 
                self.alpha * self.rho[1] * (1-self.rho[0]))
        LP = LocalStepManyDocs.calcLocalParams(
            Data, LP, alphaEbeta, alphaEbetaRem=alphaEbetaRem, **kwargs)
        assert 'resp' in LP or 'spR' in LP
        assert 'DocTopicCount' in LP
        return LP
		'''
		
    def initLPFromResp(
            self, Data, LP,
            alphaEbeta=None, alphaEbetaRem=None):
        ''' Fill in remaining local parameters given token-topic resp.

        Args
        ----
        LP : dict with fields
            * resp : 2D array, size N x K

        Returns
        -------
        LP : dict with fields
            * DocTopicCount
            * theta
            * ElogPi
        '''
        raise Exception('initLPFromResp is not implemented.')


    def get_global_suff_stats(
            self, Data, LP,
            **kwargs):

		SS = calcSummaryStats(Data, LP, **kwargs)
        return SS 


    def set_global_params(self, K=0, **kwargs):
        """ Set global parameters to provided values.
        """
        self.K = K

    def init_global_params(self, Data, K=0, **kwargs):
        """ Initialize global parameters to provided values.
        """
        self.K = K



    def calc_evidence(self, Data, SS, LP, todict=0, **kwargs):
        raise Exception('initLPFromResp is not implemented.')
        #return calcELBO(Data=Data, SS=SS, LP=LP, todict=todict,
        #                alpha=self.alpha, gamma=self.gamma,
        #                rho=self.rho, omega=self.omega,
        #                **kwargs)



def calcSummaryStats(Dslice, LP=None,
                     alpha=None,
                     alphaEbeta=None,
                     **kwargs):
    """ Calculate summary from local parameters for given data slice.

    Parameters
    -------
    Data : bnpy data object
    LP : local param dict with fields
        resp : Data.nObs x K array,
            where resp[n,k] = posterior resp of comp k
 

    Returns
    -------
    SS : SuffStatBag with K components
        Relevant fields
        * nDoc : scalar float
            Counts total documents available in provided data.
        * eta : 1D array, size K
            response parameter
        * delta : scalar float
            response dispersion parameter
    """
    raise Exception('Not implemented')
    from numpy.linalg import inv
    
    response = Data.response
    
    #theta = LP['theta']
    resp = LP['resp'] #resp[d] = resp, N_d x K
	nDocs = Data.doc_range.shape[0] - 1
	_,K = resp[0].size
	
	'''
	X = np.zeros((nDocs,K))
	
	for d in xrange(nDocs):
		respd = resp[d]
		N_d, _ = respd.size
		X[d] = (1 / float(N_d)) * np.sum(respd,axis=0)
    '''
    
    EX = np.zeros((nDocs,K)) # E[X], X[d] = \bar{Z}_d, D X K
    for d in xrange(nDocs):
		starti = Data.doc_range[d]
		endi = Data.doc_range[d+1]
		respd = resp[starti:endi,:]
		N_d, _ = respd.shape
		EX[d] = (1 / float(N_d)) * np.sum(respd,axis=0)

    
    EXTX = np.zeros((K,K)) #E[X^T X], KxK
    for d in xrange(nDocs):
		print d
		starti = Data.doc_range[d]
		endi = Data.doc_range[d+1]
		respd = resp[starti:endi,:]
		N_d, _ = respd.shape

		t1 = np.zeros(K)
		for n in xrange(N_d):
			t1+= respd[n]
			
		EXTX += (1/ float(N_d * N_d)) * np.outer(t1,t1.transpose())


    EXTXinv = inv(EXTX)  
    
    eta_update = np.dot(EXTXinv,EX.transpose())
    eta_update = np.dot(eta_update, response)

    #SS = SuffStatBag(K=K, D=Data.get_dim())
    #SS.setField('eta', eta, dims=None)
    GP = dict()
    GP['eta'] = eta_update
    
    return GP
