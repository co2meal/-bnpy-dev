import numpy as np

import OptimizerRhoOmega
from bnpy.util import NumericUtil
from bnpy.util import digamma, gammaln
from bnpy.util.StickBreakUtil import rho2beta
from bnpy.util.NumericUtil import calcRlogRdotv_allpairs
from bnpy.util.NumericUtil import calcRlogRdotv_specificpairs
from bnpy.util.NumericUtil import calcRlogR_allpairs, calcRlogR_specificpairs
from bnpy.util.NumericUtil import calcRlogR, calcRlogRdotv

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
    if isinstance(Lnon, dict):
        Llinear.update(Lnon)
        return Llinear
    return Lnon + Llinear


def calcELBO_LinearTerms(SS=None, LP=None,
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
    if LP is not None:
        nDoc = LP['theta'].shape[0]
    elif SS is not None:
        nDoc = SS.nDoc
    return L_alloc(
        nDoc=nDoc, rho=rho, omega=omega, Ebeta=Ebeta,
        alpha=alpha, gamma=gamma, todict=todict)


def calcELBO_NonlinearTerms(Data=None, SS=None, LP=None, todict=0,
                            rho=None, Ebeta=None, alpha=None,
                            resp=None, DocTopicCount=None, theta=None,
                            ElogPi=None,
                            nDoc=None, sumLogPi=None,
                            sumLogPiRem=None, sumLogPiRemVec=None,
                            Hresp=None, slackTheta=None, slackThetaRem=None,
                            gammalnTheta=None, gammalnSumTheta=None,
                            gammalnThetaRem=None,
                            thetaEmptyComp=None, ElogPiEmptyComp=None,
                            ElogPiOrigComp=None,
                            gammalnThetaOrigComp=None, slackThetaOrigComp=None,
                            returnMemoizedDict=0, **kwargs):
    """ Calculate ELBO objective terms non-linear in suff stats.
    """
    if Ebeta is None:
        Ebeta = rho2beta(rho, returnSize='K+1')

    if LP is not None:
        resp = LP['resp']
        DocTopicCount = LP['DocTopicCount']
        nDoc = DocTopicCount.shape[0]
        theta = LP['theta']
        thetaRem = LP['thetaRem']
        ElogPi = LP['ElogPi']
        ElogPiRem = LP['ElogPiRem']
        sumLogPi = np.sum(ElogPi, axis=0)
        sumLogPiRem = np.sum(ElogPiRem)
        if 'thetaEmptyComp' in LP:
            thetaEmptyComp = LP['thetaEmptyComp']
            ElogPiEmptyComp = LP['ElogPiEmptyComp']
            ElogPiOrigComp = LP['ElogPiOrigComp']
            gammalnThetaOrigComp = LP['gammalnThetaOrigComp']
            slackThetaOrigComp = LP['slackThetaOrigComp']
    elif SS is not None:
        sumLogPi = SS.sumLogPi
        nDoc = SS.nDoc
        if hasattr(SS, 'sumLogPiRemVec'):
            sumLogPiRemVec = SS.sumLogPiRemVec
        else:
            sumLogPiRem = SS.sumLogPiRem

    if DocTopicCount is not None and theta is None:
        theta = DocTopicCount + alpha * Ebeta[:-1]
        thetaRem = alpha * Ebeta[-1]

    if theta is not None and ElogPi is None:
        digammasumtheta = digamma(theta.sum(axis=1) + thetaRem)
        ElogPi = digamma(theta) - digammasumtheta[:, np.newaxis]
        ElogPiRem = digamma(thetaRem) - digammasumtheta[:, np.newaxis]

    if sumLogPi is None and ElogPi is not None:
        sumLogPi = np.sum(ElogPi, axis=0)
        sumLogPiRem = np.sum(ElogPiRem)

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

    if thetaEmptyComp is not None:
        gammalnThetaEmptyComp = nDoc * gammaln(thetaEmptyComp) - \
            gammalnThetaOrigComp
        slackThetaEmptyComp = -np.sum(thetaEmptyComp * ElogPiEmptyComp) - \
            slackThetaOrigComp

    if returnMemoizedDict:
        Mdict = dict(Hresp=Hresp,
                    slackTheta=slackTheta,
                    slackThetaRem=slackThetaRem,
                    gammalnTheta=gammalnTheta,
                    gammalnThetaRem=gammalnThetaRem,
                    gammalnSumTheta=gammalnSumTheta)
        if thetaEmptyComp is not None:
            Mdict['gammalnThetaEmptyComp'] = gammalnThetaEmptyComp
            Mdict['slackThetaEmptyComp'] = slackThetaEmptyComp
        return Mdict

    # First, compute all local-only terms
    Lentropy = Hresp.sum()
    Lslack = slackTheta.sum() + slackThetaRem
    LcDtheta = -1 * (gammalnSumTheta - gammalnTheta.sum() - gammalnThetaRem)

    # For stochastic (soVB), we need to scale up these terms
    # Only used when --doMemoELBO is set to 0 (not recommended)
    if SS is not None and SS.hasAmpFactor():
        Lentropy *= SS.ampF
        Lslack *= SS.ampF
        LcDtheta *= SS.ampF

    # Next, compute the slack term
    alphaEbeta = alpha * Ebeta
    Lslack_alphaEbeta = np.sum(alphaEbeta[:-1] * sumLogPi)
    if sumLogPiRemVec is not None:
        Ebeta_gt = 1 - np.cumsum(Ebeta[:-1])
        Lslack_alphaEbeta += alpha * np.inner(Ebeta_gt, sumLogPiRemVec)
    else:
        Lslack_alphaEbeta += alphaEbeta[-1] * sumLogPiRem
    Lslack += Lslack_alphaEbeta

    if todict:
        return dict(
            Lslack=Lslack,
            Lentropy=Lentropy,
            LcDtheta=LcDtheta)

    return LcDtheta + Lslack + Lentropy



def L_alloc(nDoc=None, rho=None, omega=None,
            alpha=None, gamma=None, todict=0, **kwargs):
    ''' Evaluate the top-level term of the surrogate objective
    '''
    K = rho.size
    eta1 = rho * omega
    eta0 = (1 - rho) * omega
    digammaBoth = digamma(eta1 + eta0)
    ElogU = digamma(eta1) - digammaBoth
    Elog1mU = digamma(eta0) - digammaBoth

    Ltop_cDiff = K * c_Beta(1, gamma) - c_Beta(eta1, eta0)
    Ltop_logpDiff = np.inner(1.0 - eta1, ElogU) + \
        np.inner(gamma - eta0, Elog1mU)

    nDoc = np.asarray(nDoc)
    if nDoc.size > 1:
        LcDsur_const = 0
        LcDsur_rhoomega = 0
        for Kd in range(nDoc.size):
            LcDsur_const += nDoc[Kd] * Kd * np.log(alpha)
            LcDsur_rhoomega += nDoc[Kd] * (np.sum(ElogU[:Kd]) + \
                np.inner(OptimizerRhoOmega.kvec(Kd), Elog1mU[:Kd]))
    else:
        LcDsur_const = nDoc * K * np.log(alpha)
        LcDsur_rhoomega = nDoc * np.sum(ElogU) + \
            nDoc * np.inner(OptimizerRhoOmega.kvec(K), Elog1mU)

    Lalloc = Ltop_cDiff + Ltop_logpDiff + LcDsur_const + LcDsur_rhoomega

    if todict:
        return dict(
            Lalloc=Lalloc,
            Lalloc_top_cDiff=Ltop_cDiff,
            Lalloc_top_logpDiff=Ltop_logpDiff,
            Lalloc_cDsur_const=LcDsur_const,
            Lalloc_cDsur_rhoomega=LcDsur_rhoomega)
    return Lalloc

def L_top(nDoc=None, rho=None, omega=None,
          alpha=None, gamma=None, todict=0, **kwargs):
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

    calpha = nDoc * K * np.log(alpha)
    cDiff = K * c_Beta(1, gamma) - c_Beta(eta1, eta0)
    return calpha + \
        cDiff + \
         + np.inner(ONcoef, ElogU) + np.inner(OFFcoef, Elog1mU)


def calcHrespForMergePairs(resp, Data, mPairIDs, returnVec=1):
    ''' Calculate resp entropy terms for all candidate merge pairs

    Returns
    ---------
    Hresp : 2D array, size K x K
    or 
    Hresp : 1D array, size M
        where each entry corresponds to one merge pair in mPairIDs
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
    if returnVec:
        Hvec = np.zeros(len(mPairIDs))
        for ii, (kA, kB) in enumerate(mPairIDs):
            Hvec[ii] = -1 * Hmat[kA, kB]
        return Hvec
    else:
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


def calcELBO_FixedDocTopicCountIgnoreEntropy(
        alpha=None, gamma=None,
        rho=None, omega=None,
        DocTopicCount=None):
    K = rho.size
    Hresp = np.zeros(K)
    Lnon = calcELBO_NonlinearTerms(
        nDoc=DocTopicCount.shape[0],
        DocTopicCount=DocTopicCount, alpha=alpha,
        rho=rho, omega=omega, Hresp=Hresp)
    Llinear = calcELBO_LinearTerms(alpha=alpha, gamma=gamma,
                                   rho=rho, omega=omega,
                                   nDoc=DocTopicCount.shape[0])
    return Lnon + Llinear


def calcMergeTermsFromSeparateLP(
        Data=None,
        LPa=None, SSa=None,
        LPb=None, SSb=None,
        mUIDPairs=None):
    ''' Compute merge terms that combine two comps from separate LP dicts.
    
    Returns
    -------
    Mdict : dict of key, array-value pairs
    '''
    M = len(mUIDPairs)
    m_sumLogPi = np.zeros(M)
    m_gammalnTheta = np.zeros(M)
    m_slackTheta = np.zeros(M)
    m_Hresp = np.zeros(M)

    assert np.allclose(LPa['digammaSumTheta'], LPb['digammaSumTheta'])
    for m, (uidA, uidB) in enumerate(mUIDPairs):
        kA = SSa.uid2k(uidA)
        kB = SSb.uid2k(uidB)

        m_resp = LPa['resp'][:, kA] + LPb['resp'][:, kB]
        if hasattr(Data, 'word_count'):
            m_Hresp[m] = -1 * calcRlogRdotv(
                m_resp[:,np.newaxis], Data.word_count)
        else:
            m_Hresp[m] = -1 * calcRlogR(m_resp[:,np.newaxis])

        DTC_vec = LPa['DocTopicCount'][:, kA] + LPb['DocTopicCount'][:, kB]
        theta_vec = LPa['theta'][:, kA] + LPb['theta'][:, kB]
        m_gammalnTheta[m] = np.sum(gammaln(theta_vec))
        ElogPi_vec = digamma(theta_vec) - LPa['digammaSumTheta']
        m_sumLogPi[m] = np.sum(ElogPi_vec)
        # slack = (Ndm - theta_dm) * E[log pi_dm]
        slack_vec = ElogPi_vec
        slack_vec *= (DTC_vec - theta_vec)
        m_slackTheta[m] = np.sum(slack_vec)
    return dict(
        Hresp=m_Hresp,
        gammalnTheta=m_gammalnTheta,
        slackTheta=m_slackTheta,
        sumLogPi=m_sumLogPi)

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
