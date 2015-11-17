'''
FiniteMMSB.py
'''

import numpy as np

from scipy.sparse import csc_matrix
from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import gammaln, digamma, EPS

from bnpy.util import StickBreakUtil
from bnpy.allocmodel.topics import OptimizerRhoOmegaBetter
from bnpy.allocmodel.topics.HDPTopicUtil import c_Beta, c_Dir, L_top

from FiniteMMSB import FiniteMMSB

class HDPMMSB(FiniteMMSB):

    """ Mixed membership stochastic block model, with nonparametric HDP prior.

    Attributes
    -------
    inferType : string {'EM', 'VB', 'moVB', 'soVB'}
        indicates which updates to perform for local/global steps
    K : int
        number of components
    alpha : float
        scalar symmetric Dirichlet prior on mixture weights
    gamma : float
        scalar concentration for top-level Dirichlet process prior

    Attributes for VB
    ---------
    theta : 2D array, nNodes x K
        theta[n,:] gives parameters for Dirichlet variational factor
        defining distribution over membership probabilities for node n

    rho : 1D array, size K
    omega : 1D array, size K
    """

    def __init__(self, inferType, priorDict=dict()):
        if inferType.count('EM') > 0:
            raise NotImplementedError(
                'EM not implemented for FiniteMMSB (yet)')

        self.inferType = inferType
        self.set_prior(**priorDict)
        self.K = 0

    def set_prior(self, alpha=.1, gamma=10.0):
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def get_active_comp_probs(self):
        print 'TODO'

    def getCompDims(self):
        ''' Get dimensions of latent component interactions.

        Overrides default of ('K',), since E_log_soft_ev needs to be ('K','K')

        Returns
        -------
        dims : tuple
        '''
        return ('K', 'K',)

    def E_logPi(self, returnRem=0):
        ''' Compute expected value of log \pi for each node and state.

        Returns
        -------
        ElogPi : 2D array, nNodes x K
        '''
        digammasumtheta = digamma(
            self.theta.sum(axis=1) + self.thetaRem)
        ElogPi = digamma(self.theta) - digammasumtheta[:, np.newaxis]
        if returnRem:
            ElogPiRem = digamma(self.thetaRem) - digammasumtheta
            return ElogPi, ElogPiRem
        return ElogPi

    # calc_local_params inherited from FiniteMMSB
    # get_global_suff_stats inherited from FiniteMMSB

    def forceSSInBounds(self, SS):
        ''' Force certain fields in bounds, to avoid numerical issues.

        Returns
        -------
        Nothing.  SS is updated in-place.
        '''
        np.maximum(SS.NodeStateCount, 0, out=SS.NodeStateCount)

    def update_global_params_VB(self, SS, **kwargs):
        ''' Update global parameter theta to optimize VB objective.

        Post condition
        --------------
        Attributes rho,omega,theta set to optimal value given suff stats.
        '''
        nGlobalIters = 2
        nNode = SS.NodeStateCount.shape[0]

        if not hasattr(self, 'rho') or self.rho.size != SS.K:
            self.rho = OptimizerRhoOmegaBetter.make_initrho(
                SS.K, nNode, self.gamma)
        self.omega = OptimizerRhoOmegaBetter.make_initomega(
            SS.K, nNode, self.gamma)
        # Update theta with recently updated info from suff stats
        self.theta, self.thetaRem = updateThetaAndThetaRem(
            SS, rho=self.rho, alpha=self.alpha, gamma=self.gamma)
        for giter in xrange(nGlobalIters):
            self.rho, self.omega = updateRhoOmega(
                theta=self.theta, thetaRem=self.thetaRem,
                initrho=self.rho, omega=self.omega, 
                alpha=self.alpha, gamma=self.gamma)
            self.theta, self.thetaRem = updateThetaAndThetaRem(
                SS, rho=self.rho, alpha=self.alpha, gamma=self.gamma)
        
    def set_global_params(self, hmodel=None,
                          rho=None, omega=None, theta=None, thetaRem=None,
                          **kwargs):
        ''' Set rho, omega, theta to specific provided values.
        '''
        if hmodel is not None:
            self.K = hmodel.allocModel.K
            if hasattr(hmodel.allocModel, 'rho'):
                self.rho = hmodel.allocModel.rho
                self.omega = hmodel.allocModel.omega
            else:
                raise AttributeError('Unrecognized hmodel. No field rho.')
            if hasattr(hmodel.allocModel, 'theta'):
                self.theta = hmodel.allocModel.theta
                self.thetaRem = hmodel.allocModel.thetaRem
            else:
                raise AttributeError('Unrecognized hmodel. No field theta.')
        elif rho is not None \
                and omega is not None \
                and theta is not None:
            self.rho = rho
            self.omega = omega
            self.theta = theta
            self.thetaRem = thetaRem
            self.K = omega.size
        else:
            self._set_global_params_from_scratch(**kwargs)

    def _set_global_params_from_scratch(self, beta=None,
                                        Data=None, nNodes=None, **kwargs):
        ''' Set rho, omega to values that reproduce provided appearance probs

        Args
        --------
        beta : 1D array, size K
            beta[k] gives top-level probability for active comp k
        '''
        if nNodes is None:
            nNodes = Data.nNodes
        if nNodes is None:
            raise ValueError('Bad parameters. nNodes not specified.')
        if beta is None:
            raise ValueError('Bad parameters. Vector beta not specified.')
        beta = beta / beta.sum()
        Ktmp = beta.size
        rem = np.minimum(0.05, 1. / (Ktmp))
        beta = np.hstack([np.squeeze(beta), rem])
        beta = beta / np.sum(beta)
        self.K = beta.size - 1
        self.rho, self.omega = _beta2rhoomega(beta, nNodes)
        assert self.rho.size == self.K
        assert self.omega.size == self.K


    def init_global_params(self, Data, K=0, **kwargs):
        ''' Initialize global parameters "from scratch" to reasonable values.

        Post condition
        --------------
        Global parameters rho, omega, theta, thetaRem set to
        valid values.
        '''
        self.K = K
        initbeta = (1.0 - 0.01)/K * np.ones(K)
        assert np.sum(initbeta) < 1.0
        self.rho, self.omega = _beta2rhoomega(
            beta=initbeta, K=K, 
            nDoc=Data.nNodes, gamma=self.gamma)
        self.theta = self.alpha * np.tile(initbeta, (Data.nNodes, 1))
        self.thetaRem = self.alpha * (1 - initbeta.sum())

    def calc_evidence(self, Data, SS, LP, todict=0, **kwargs):
        ''' Compute training objective function on provided input.

        Returns
        -------
        L : scalar float
        '''
        Lalloc = self.L_alloc_no_slack()
        Lslack = self.L_slack(SS)
        if SS.hasELBOTerm('Hresp'):
            Lentropy = SS.getELBOTerm('Hresp')
        else:
            # L_entropy function inherited from FiniteMMSB
            Lentropy = self.L_entropy(LP)
        if todict:
            return dict(Lentropy=Lentropy, Lalloc=Lalloc, Lslack=Lslack)
        return Lalloc + Lentropy + Lslack

    def L_alloc_no_slack(self):
        ''' Compute allocation term of objective function, without slack term

        Returns
        -------
        L : scalar float
        '''
        prior_cDir = L_top(nDoc=self.theta.shape[0], 
            alpha=self.alpha, gamma=self.gamma,
            rho=self.rho, omega=self.omega)
        post_cDir = c_Dir(self.theta, self.thetaRem)
        return prior_cDir - post_cDir

    def L_slack(self, SS):
        ''' Compute slack term of the allocation objective function.

        Returns
        -------
        L : scalar float
        '''
        ElogPi, ElogPiRem = self.E_logPi(returnRem=1)
        Ebeta = StickBreakUtil.rho2beta(self.rho, returnSize='K')
        Q = SS.NodeStateCount + self.alpha * Ebeta - self.theta
        Lslack = np.sum(Q * ElogPi)

        alphaEbetaRem = self.alpha * (1.0 - Ebeta.sum())
        LslackRem = np.sum((alphaEbetaRem - self.thetaRem) * ElogPiRem)
        return Lslack + LslackRem

    def to_dict(self):
        return dict(theta=self.theta, rho=self.rho, omega=self.omega)

    def from_dict(self, myDict):
        self.inferType = myDict['inferType']
        self.K = myDict['K']
        self.theta = myDict['theta']
        self.rho = myDict['rho']
        self.omega = myDict['omega']

    def get_prior_dict(self):
        return dict(alpha=self.alpha, gamma=self.gamma)




def updateThetaAndThetaRem(SS, rho=None, alpha=1.0, gamma=10.0):
    ''' Update parameters theta to maximize objective given suff stats.

    Returns
    ---------
    theta : 2D array, nNodes x K
    thetaRem : scalar
    '''
    K = SS.K
    nNodes = SS.NodeStateCount.shape[0]
    if rho is None or rho.size != K:
        rho = OptimizerRhoOmegaBetter.make_initrho(K, nNodes, gamma)

    # Calculate E_q[alpha * Beta_l] for l = 1, ..., K+1
    Ebeta = StickBreakUtil.rho2beta(rho, returnSize='K')
    alphaEbeta = alpha * Ebeta
    alphaEbetaRem = alpha * (1- Ebeta.sum())

    theta = alphaEbeta + SS.NodeStateCount
    thetaRem = alphaEbetaRem
    return theta, thetaRem

def _beta2rhoomega(beta, K, nDoc=10, gamma=10):
    ''' Find vectors rho, omega that are probable given beta

    Returns
    --------
    rho : 1D array, size K
    omega : 1D array, size K
    '''
    assert beta.size == K or beta.size == K + 1
    rho = OptimizerRhoOmegaBetter.beta2rho(beta, K)
    omega = OptimizerRhoOmegaBetter.make_initomega(K, nDoc, gamma)
    return rho, omega

def updateRhoOmega(
        theta=None, thetaRem=None,
        initrho=None,
        omega=None,
        alpha=0.5, gamma=10,
        logFunc=None):
    ''' Update rho, omega via numerical optimization.

    Will set vector omega to reasonable fixed value,
    and do gradient descent to optimize the vector rho.

    Returns
    -------
    rho : 1D array, size K
    omega : 1D array, size K
    '''
    nDoc = theta.shape[0]
    K = theta.shape[1]
    # Verify initial rho
    assert initrho is not None
    assert initrho.size == K
    # Verify initial omega
    assert omega is not None
    assert omega.size == K
    # Compute summaries of theta needed to update rho
    # sumLogPi : 1D array, size K
    # sumLogPiRem : scalar
    digammasumtheta = digamma(theta.sum(axis=1) + thetaRem)
    ElogPi = digamma(theta) - digammasumtheta[:, np.newaxis]
    sumLogPi = np.sum(ElogPi, axis=0)
    ElogPiRem = digamma(thetaRem) - digammasumtheta
    sumLogPiRem = np.sum(ElogPiRem)
    # Do the optimization
    try:
        rho, omega, fofu, Info = \
            OptimizerRhoOmegaBetter.find_optimum_multiple_tries(
                nDoc=nDoc,
                sumLogPiActiveVec=sumLogPi,
                sumLogPiRem=sumLogPiRem,
                gamma=gamma,
                alpha=alpha,
                initrho=initrho,
                initomega=omega,
                do_grad_omega=0,
                do_grad_rho=1)
    except ValueError as error:
        if logFunc:
            logFunc('***** Rho optim failed. Remain at cur val. ' + \
                str(error))
        rho = initrho

    assert rho.size == K
    assert omega.size == K
    return rho, omega
