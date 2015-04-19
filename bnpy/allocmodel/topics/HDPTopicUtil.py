import numpy as np

import OptimizerRhoOmega
from bnpy.util import NumericUtil
from bnpy.util import digamma, gammaln
from bnpy.util.StickBreakUtil import rho2beta

ELBOTermDimMap = dict(
    slackTheta='K',
    slackThetaRem=None,
    gammalnTheta='K',
    gammalnThetaRem=None,
    gammalnSumTheta=None,
    Hresp=None,
    )

def calcELBO(**kwargs):
    """ Calculate ELBO objective for provided model state.

    Returns
    -------
    L : scalar float
        L is the value of the objective function at provided state.
    """
    Llinear = calcELBO_LinearTerms(**kwargs)
    Lnon = calcELBO_NonlinearTerms(**kwargs)
    return Lnon + Llinear

def calcELBO_LinearTerms(SS=None,
        nDoc=None,
        rho=None, omega=None, Ebeta=None,
        alpha=0, gamma=None, 
        afterGlobalStep=0, todict=0, **kwargs):
    """ Calculate ELBO objective terms that are linear in suff stats.

    Returns
    -------
    L : scalar float
        L is sum of any term in ELBO that is const/linear wrt suff stats.
    """
    if SS is not None:
        nDoc = SS.nDoc
    return L_top(nDoc=nDoc,
        rho=rho, omega=omega, Ebeta=Ebeta,
        alpha=alpha, gamma=gamma)

def calcELBO_NonlinearTerms(Data=None, SS=None, LP=None,
        rho=None, Ebeta=None, alpha=None,
        resp=None, DocTopicCount=None, theta=None, ElogPi=None,
        nDoc=None, sumLogPi=None, sumLogPiRem=None,
        Hresp=None, slackTheta=None, slackThetaRem=None,
        gammalnTheta=None, gammalnSumTheta=None, gammalnThetaRem=None,
        returnMemoizedDict=0, **kwargs):
    """ Calculate ELBO objective terms non-linear in suff stats.
    """
    if SS is not None:
        sumLogPi = SS.sumLogPi
        sumLogPiRem = SS.sumLogPiRem
    
    if LP is not None:
        resp = LP['resp']
        DocTopicCount = LP['DocTopicCount']
        theta = LP['theta']
        thetaRem = LP['thetaRem']
        ElogPi = LP['ElogPi']
        ElogPiRem = LP['ElogPiRem']

    if Hresp is None:
        if SS is not None and SS.hasELBOTerm('Hresp'):
            Hresp = SS.getELBOTerm('Hresp')
        else:
            if hasattr(Data, 'word_count'):
                Hresp = -1 * NumericUtil.calcRlogRdotv(resp, Data.word_count)
            else:
                Hresp = -1 * NumericUtil.calcRlogR(resp)

    if slackTheta is None:
        if SS is not None and SS.hasELBOTerm('slackTheta'):
            slackTheta = SS.getELBOTerm('slackTheta')
            slackThetaRem = SS.getELBOTerm('slackThetaRem')
        else:
            slackTheta = DocTopicCount - theta
            slackTheta *= ElogPi
            slackTheta = np.sum(slackTheta, axis=0)
            slackThetaRem = -1 * np.sum(thetaRem * ElogPiRem)

    if gammalnTheta is None:
        if SS is not None and SS.hasELBOTerm('gammalnTheta'):
            gammalnSumTheta = SS.getELBOTerm('gammalnSumTheta')
            gammalnTheta = SS.getELBOTerm('gammalnTheta')
            gammalnThetaRem = SS.getELBOTerm('gammalnThetaRem')
        else:
            sumTheta = np.sum(theta, axis=1) + thetaRem
            gammalnSumTheta = np.sum(gammaln(sumTheta))
            gammalnTheta = np.sum(gammaln(theta), axis=0)
            gammalnThetaRem = theta.shape[0] * gammaln(thetaRem)

    if returnMemoizedDict:
        return dict(Hresp=Hresp,
            slackTheta=slackTheta,
            slackThetaRem=slackThetaRem,
            gammalnTheta=gammalnTheta,
            gammalnThetaRem=gammalnThetaRem,
            gammalnSumTheta=gammalnSumTheta)

    # First, compute all local-only terms
    Lentropy = Hresp.sum()
    Lslack = slackTheta.sum() + slackThetaRem
    LcDtheta =  -1 * (gammalnSumTheta - gammalnTheta.sum() - gammalnThetaRem)
    
    # For stochastic (soVB), we need to scale up these terms
    # Only used when --doMemoELBO is set to 0 (not recommended)
    if SS is not None and SS.hasAmpFactor():
        Lentropy *= SS.ampF
        Lslack *= SS.ampF
        LcDtheta *= SS.ampF

    # Next, compute the slack term 
    if Ebeta is None:
        Ebeta = rho2beta(rho, returnSize='K+1')
    alphaEbeta = alpha * Ebeta
    Lslack_alphaEbeta = np.sum(alphaEbeta[:-1] * sumLogPi) \
                        + alphaEbeta[-1] * sumLogPiRem
    Lslack += Lslack_alphaEbeta

    return LcDtheta + Lslack + Lentropy 


def L_top(nDoc=None, rho=None, omega=None, 
        alpha=None, gamma=None, **kwargs):
    ''' Evaluate the top-level term of the surrogate objective
    '''
    K = rho.size
    eta1 = rho * omega
    eta0 = (1 - rho) * omega
    digammaBoth = digamma(eta1 + eta0)
    ElogU = digamma(eta1) - digammaBoth
    Elog1mU = digamma(eta0) - digammaBoth

    ONcoef = nDoc + 1.0 - eta1
    OFFcoef = nDoc * OptimizerRhoOmega.kvec(K) + gamma - eta0

    calpha = nDoc * K  * np.log(alpha)
    cDiff = K * c_Beta(1, gamma) - c_Beta(eta1, eta0)

    return calpha + \
        cDiff + \
        np.inner(ONcoef, ElogU) + np.inner(OFFcoef, Elog1mU)




def calcHrespForMergePairs(resp, Data, mPairIDs):
    ''' Calculate resp entropy terms for all candidate merge pairs

    Returns
    ---------
    Hresp : 2D array, size K x K
    '''
    if hasattr(Data, 'word_count'):
        if mPairIDs is None:
            Hmat = calcRlogRdotv_allpairs(resp, Data.word_count)
        else:
            Hmat = calcRlogRdotv_specificpairs(resp, Data.word_count, mPairIDs)
    else:
        if mPairIDs is None:
            Hmat = calcRlogR_allpairs(resp)
        else:
            Hmat = calcRlogR_specificpairs(resp, mPairIDs)
    return -1 * Hmat


def c_Beta(a1, a0):
    ''' Evaluate cumulant function of the Beta distribution

    When input is vectorized, we compute sum over all entries.

    Returns
    -------
    c : scalar real
    '''
    return np.sum(gammaln(a1 + a0)) - np.sum(gammaln(a1)) - np.sum(gammaln(a0))


def c_Dir(AMat, arem=None):
    ''' Evaluate cumulant function of the Dir distribution

    When input is vectorized, we compute sum over all entries.

    Returns
    -------
    c : scalar real
    '''
    AMat = np.asarray(AMat)
    D = AMat.shape[0]
    if arem is None:
        if AMat.ndim == 1:
            return gammaln(np.sum(AMat)) - np.sum(gammaln(AMat))
        else:
            return np.sum(gammaln(np.sum(AMat, axis=1))) \
                - np.sum(gammaln(AMat))

    return np.sum(gammaln(np.sum(AMat, axis=1) + arem)) \
        - np.sum(gammaln(AMat)) \
        - D * np.sum(gammaln(arem))


def E_cDalphabeta_surrogate(alpha, rho, omega):
    ''' Compute expected value of cumulant function of alpha * beta.

    Returns
    -------
    csur : scalar float
    '''
    K = rho.size
    eta1 = rho * omega
    eta0 = (1 - rho) * omega
    digammaBoth = digamma(eta1 + eta0)
    ElogU = digamma(eta1) - digammaBoth
    Elog1mU = digamma(eta0) - digammaBoth
    OFFcoef = OptimizerRhoOmega.kvec(K)
    calpha = gammaln(alpha) + (K + 1) * np.log(alpha)
    return calpha + np.sum(ElogU) + np.inner(OFFcoef, Elog1mU)

"""
def E_cDir_alphabeta__Numeric(self):
''' Numeric integration of the expectation
'''
g1 = self.rho * self.omega
g0 = (1 - self.rho) * self.omega
assert self.K <= 2
if self.K == 1:
    us = np.linspace(1e-14, 1 - 1e-14, 1000)
    logpdf = gammaln(g1 + g0) - gammaln(g1) - gammaln(g0) \
        + (g1 - 1) * np.log(us) + (g0 - 1) * np.log(1 - us)
    pdf = np.exp(logpdf)
    b1 = us
    bRem = 1 - us
    Egb1 = np.trapz(gammaln(self.alpha * b1) * pdf, us)
    EgbRem = np.trapz(gammaln(self.alpha * bRem) * pdf, us)
    EcD = gammaln(self.alpha) - Egb1 - EgbRem
return EcD

def E_cDir_alphabeta__MonteCarlo(self, S=1000, seed=123):
''' Monte Carlo approximation to the expectation
'''
PRNG = np.random.RandomState(seed)
g1 = self.rho * self.omega
g0 = (1 - self.rho) * self.omega
cD_abeta = np.zeros(S)
for s in range(S):
    u = PRNG.beta(g1, g0)
    u = np.minimum(np.maximum(u, 1e-14), 1 - 1e-14)
    beta = np.hstack([u, 1.0])
    beta[1:] *= np.cumprod(1.0 - u)
    cD_abeta[s] = gammaln(
        self.alpha) - gammaln(self.alpha * beta).sum()
return np.mean(cD_abeta)

def E_cDir_alphabeta__Surrogate(self):
calpha = gammaln(self.alpha) + (self.K + 1) * np.log(self.alpha)

g1 = self.rho * self.omega
g0 = (1 - self.rho) * self.omega
digammaBoth = digamma(g1 + g0)
ElogU = digamma(g1) - digammaBoth
Elog1mU = digamma(g0) - digammaBoth
OFFcoef = OptimizerRhoOmega.kvec(self.K)
cRest = np.sum(ElogU) + np.inner(OFFcoef, Elog1mU)

return calpha + cRest
"""