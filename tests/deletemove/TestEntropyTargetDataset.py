import numpy as np
import unittest
import sys

try:
    from matplotlib import pylab
    doViz = True
except ImportError:
    doViz = False

rEPS = 1e-40

def makeNewResp_Exact(resp, rule='renorm'):
    """ Create new resp matrix that exactly obeys required constraints.
    """
    respNew = resp[:, 1:].copy()
    if rule == 'renorm':
        respNew /= respNew.sum(axis=1)[:,np.newaxis]
    elif rule == 'minH':
        mink = np.sum(respNew * np.log(respNew),axis=0).argmin()
        print '>>> mink=', mink
        respNew[:, mink] += resp[:, 0]
    respNew = np.maximum(respNew, rEPS)
    return respNew

def makeNewResp_Approx(resp):
    """ Create new resp matrix that exactly obeys required constraints.
    """
    respNew = resp[:, 1:].copy()
    respNew = np.maximum(respNew, rEPS)
    return respNew

def calcRlogR(R):
    """
    
    Returns
    -------
    H : 2D array, size N x K
        each entry is positive.
    """
    return -1 * R * np.log(R)

class TestK3(unittest.TestCase):

    def shortDescription(self):
        return None

    def setUp(self, K=3, N=100, dtargetMinResp=0.01, rule='renorm',
                    Rsource='random'):
        rng = np.random.RandomState(101)

        if Rsource == 'random':
            R = 1.0/(K-1) + dtargetMinResp + rng.rand(N, K)
            R[:, 0] = dtargetMinResp
            assert R.sum(axis=1).min() > 1.0
        elif Rsource == 'toydata':
            raise NotImplementedError('TODO')
            # Run bnpy for a few iters on toy data to get "realistic" 
            # responsibility matrix from a junk-y initialization.

        R = np.maximum(R, rEPS)
        R /= R.sum(axis=1)[:,np.newaxis]
        assert np.all(R[:,0] <= dtargetMinResp)

        self.K = K
        self.R = R
        self.Rnew_Exact = makeNewResp_Exact(R, rule=rule)
        self.Rnew_Approx = makeNewResp_Approx(R)

    def test_entropy_gt_zero(self):
        """ Verify that all entropy calculations yield positive values.
        """
        H = calcRlogR(self.R)
        Hnew_exact = calcRlogR(self.Rnew_Exact)
        Hnew_approx = calcRlogR(self.Rnew_Approx)

        assert np.all(H > -1e-10)
        assert np.all(Hnew_exact > -1e-10)
        assert np.all(Hnew_approx > -1e-10)

    def test_entropy_drops_from_old_to_new(self):
        """ Verify that entropy of original is higher than candidate
        """
        H = np.sum(calcRlogR(self.R), axis=1)
        Hnew_exact = np.sum(calcRlogR(self.Rnew_Exact), axis=1)
        assert np.all(H > Hnew_exact)

    def test_plot_entropy_vs_rVals(self):
        if not doViz:
            self.skipTest("Required module matplotlib unavailable.")
        H = np.sum(calcRlogR(self.R))
        Hnew_exact = np.sum(calcRlogR(self.Rnew_Exact))
        Hnew_avec = np.sum(calcRlogR(self.Rnew_Approx), axis=0)
        mink = Hnew_avec.argmin()
        Hnew_approx = Hnew_avec.sum() - Hnew_avec[mink]

        np.set_printoptions(precision=4, suppress=True, linewidth=100)
        print ''
        print '--- R original'
        print self.R[:3, :10]
        print self.R[-3:,:10]

        print '--- R proposal'
        print self.Rnew_Exact[:3, :10]
        print self.Rnew_Exact[-3:, :10]

        print '--- H original'
        print H
        print '--- H proposal exact'
        print Hnew_exact
        print '--- H proposal lowerbound'
        print Hnew_approx
        print '--- H proposal approx'
        print Hnew_approx + Hnew_avec[mink]


class TestK10(TestK3):
    def setUp(self):
        super(TestK10, self).setUp(K=10, rule='minH')

