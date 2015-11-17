'''
FiniteAssortativeMMSB.py

Assortative mixed membership stochastic blockmodel.
'''
import numpy as np
import itertools

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import gammaln, digamma, EPS

from bnpy.util import StickBreakUtil
from bnpy.allocmodel.topics import OptimizerRhoOmega
from bnpy.allocmodel.topics.HDPTopicUtil import c_Beta, c_Dir, L_top

from FiniteAssortativeMMSB import FiniteAssortativeMMSB
from HDPMMSB import updateRhoOmega, updateThetaAndThetaRem

class HDPAssortativeMMSB(FiniteAssortativeMMSB):

    """ Assortative version of MMSB, with HDP prior.

    Attributes
    -------
    * inferType : string {'EM', 'VB', 'moVB', 'soVB'}
        indicates which updates to perform for local/global steps
    * K : int
        number of components
    * alpha : float
        scalar symmetric Dirichlet prior on mixture weights

    Attributes for VB
    ---------
    * theta : 1D array, size K
        Estimated parameters for Dirichlet posterior over mix weights
        theta[k] > 0 for all k
    """

    def __init__(self, inferType, priorDict=dict()):
        super(HDPAssortativeMMSB, self).__init__(inferType, priorDict)

    def set_prior(self, alpha=0.5, gamma=10, epsilon=0.05):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def getCompDims(self):
        ''' Get dimensions of latent component interactions.

        Assortative models use only K states.

        Returns
        -------
        dims : tuple
        '''
        return ('K',)

    def E_logPi(self, returnRem=0):
        ''' Compute expected probability \pi for each node and state

        Returns
        -------
        ElogPi : nNodes x K
        '''
        digammasumtheta = digamma(
            self.theta.sum(axis=1) + self.thetaRem)
        ElogPi = digamma(self.theta) - digammasumtheta[:, np.newaxis]
        if returnRem:
            ElogPiRem = digamma(self.thetaRem) - digammasumtheta
            return ElogPi, ElogPiRem
        return ElogPi

    # calc_local_params inherited from FiniteAssortativeMMSB
    # get_global_suff_stats inherited from FiniteAssortativeMMSB
 

    def update_global_params_VB(self, SS, **kwargs):
        ''' Update global parameter theta to optimize VB objective.

        Post condition
        --------------
        Attributes rho,omega,theta set to optimal value given suff stats.
        '''
        nGlobalIters = 2

        if not hasattr(self, 'rho'):
            self.rho = OptimizerRhoOmega.create_initrho(SS.K)
        if not hasattr(self, 'omega'):
            nDoc = SS.NodeStateCount.shape[0]
            self.omega = (nDoc + self.gamma) * np.ones(SS.K)

        # Update theta with recently updated info from suff stats
        self.theta, self.thetaRem = updateThetaAndThetaRem(
            SS, alpha=self.alpha, rho=self.rho)

        for giter in xrange(nGlobalIters):
            self.rho, self.omega = updateRhoOmega(
                theta=self.theta, thetaRem=self.thetaRem,
                initrho=self.rho, initomega=self.omega, 
                alpha=self.alpha, gamma=self.gamma)

            self.theta, self.thetaRem = updateThetaAndThetaRem(
                SS, alpha=self.alpha, rho=self.rho)

        
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
        self.rho, self.omega = self._beta2rhoomega(beta, nNodes)
        assert self.rho.size == self.K
        assert self.omega.size == self.K

    def _beta2rhoomega(self, beta, nDoc=10):
        ''' Find vectors rho, omega that are probable given beta

        Returns
        --------
        rho : 1D array, size K
        omega : 1D array, size K
        '''
        assert abs(np.sum(beta) - 1.0) < 0.001
        rho = OptimizerRhoOmega.beta2rho(beta, self.K)
        omega = (nDoc + self.gamma) * np.ones(rho.size)
        return rho, omega

    def init_global_params(self, Data, K=0, **kwargs):
        ''' Initialize global parameters "from scratch" to reasonable values.

        Post condition
        --------------
        Attributes theta, K set to reasonable values.
        '''
        self.K = K
        PRNG = np.random.RandomState(K)
        initNodeStateCount = PRNG.rand(Data.nNodes, K)
        self.theta = self.alpha + initNodeStateCount

        self.rho = OptimizerRhoOmega.create_initrho(K)
        self.omega = (1.0 + self.gamma) * np.ones(K)

        Ebeta = StickBreakUtil.rho2beta(self.rho, returnSize='K')
        self.thetaRem = self.alpha * (1 - Ebeta.sum())

    def calc_evidence(self, Data, SS, LP, todict=0, **kwargs):
        ''' Compute training objective function on provided input.

        Returns
        -------
        L : scalar float
        '''
        Lalloc = self.L_alloc_no_slack()
        Lslack = self.L_slack(SS)
        # Compute entropy term
        if SS.hasELBOTerm('Hresp_fg'):
            Lentropy = SS.getELBOTerm('Hresp_fg').sum() + \
                SS.getELBOTerm('Hresp_bg')
        else:
            Lentropy = self.L_entropy(LP)

        if SS.hasELBOTerm('Ldata_bg'):
            Lbgdata = SS.getELBOTerm('Ldata_bg')
        else:
            Lbgdata = LP['Ldata_bg']
        if todict:
            return dict(Lentropy=Lentropy, 
                Lalloc=Lalloc, Lslack=Lslack,
                Lbgdata=Lbgdata)
        return Lalloc + Lentropy + Lslack + Lbgdata


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



