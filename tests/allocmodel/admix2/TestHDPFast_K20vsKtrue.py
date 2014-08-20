import unittest
import numpy as np

from DebugUtil import np2flatstr
from bnpy.allocmodel.admix2.HDPFast import HDPFast
import bnpy.allocmodel.admix2.OptimizerHDPFast as Optim
from bnpy.suffstats.SuffStatBag import SuffStatBag

Qtrue = np.load('Ktrue_input.npz')
Q20 = np.load('K20_input.npz')

grabIDs = 0

class Test(unittest.TestCase):

  def shortDescription(self):
    return None

  def setUp(self):
    ''' Identify permutation of K20 ids to exactly match the "true" order
    ''' 

    ## Choice of alpha can make the difference between 
    ## whether the objective prefers Ktrue or Kmany
    self.alpha = 0.5 # 0.99 --> bad
    self.gamma = 10

    DTCtrue = Qtrue['DocTopicCount'].copy()
    DTC20 = Q20['DocTopicCount'].copy()
    Ntrue = np.sum(DTCtrue, axis=0)
    N20 = np.sum(DTC20, axis=0)

    mask =  N20 > 0.01
    N8 = N20[mask]
    DTC8 = DTC20[:, mask]

    ## Align the columns
    sortIDs = np.zeros(8, dtype=np.int32)
    for k in range(8):
      sortIDs[k] = np.argmin( np.abs( N8 - Ntrue[k]))
    self.grabIDs = np.flatnonzero(mask)[sortIDs]
    self.N20 = N20
    self.Ntrue = Ntrue
    self.DTCtrue = DTCtrue
    self.DTC20 = DTC20

    TrueArgs = dict(
      alpha = Qtrue['alpha'],
      gamma = Qtrue['gamma'],
      nDoc = Qtrue['nDoc'],
      DocTopicCount = Qtrue['DocTopicCount'],
      )
    self.rho_true, self.omega_true, self.f_true, self.Info_true \
        = Optim.find_optimum_multiple_tries(**TrueArgs)

    K20Args = dict(
      alpha = Q20['alpha'],
      gamma = Q20['gamma'],
      nDoc = Q20['nDoc'],
      DocTopicCount = Q20['DocTopicCount'],
      )
    rho20, omega20, f20, Info = Optim.find_optimum_multiple_tries(
                                      **K20Args)
    self.rho20=rho20
    self.omega20=omega20
    self.f20 = f20
    self.Info20 = Info

  def test_calc_evidence(self):
    ''' Make HDPFast object and evaluate evidence
    '''
    print ''
    mtrue = HDPFast('VB', dict(alpha=self.alpha, gamma=self.gamma))
    mtrue.rho = self.rho_true
    mtrue.omega = self.omega_true
    mtrue.K = mtrue.rho.size

    SS = SuffStatBag(K=mtrue.K, D=2, A=5)
    SS.setField('nDoc', 5.0, dims=None)
    SS.setField('DocTopicCount', Qtrue['DocTopicCount'], dims=('A', 'K'))
    SS.setELBOTerm('ElogqZ', np.zeros(8), dims='K')

    ELBOtrue = mtrue.calc_evidence(None, SS, None)
    print '%.6f' % (ELBOtrue)


    m20 = HDPFast('VB', dict(alpha=Qtrue['alpha'], gamma=Qtrue['gamma']))
    m20.rho = self.rho20
    m20.omega = self.omega20
    m20.K = m20.rho.size

    SS20 = SuffStatBag(K=m20.K, D=2, A=5)
    SS20.setField('nDoc', 5.0, dims=None)
    SS20.setField('DocTopicCount', Q20['DocTopicCount'], dims=('A', 'K'))
    SS20.setELBOTerm('ElogqZ', np.zeros(20), dims='K')

    ELBO20 = m20.calc_evidence(None, SS20, None)
    print '%.6f' % (ELBO20)

    print (ELBOtrue - ELBO20) 
    print -1 * (self.f_true - self.f20) * SS.nDoc

    print 'MISSING GAP TERM (D K log alpha)'
    print '%.6f' % (SS.nDoc * (SS.K - SS20.K) * np.log(self.alpha))


  def test_inputs_equal_in_nonzero_columns(self):
    print ''

    N8 = self.N20[self.grabIDs]
    print np2flatstr(self.Ntrue)
    print np2flatstr(N8)
    assert np.allclose(self.Ntrue, N8)

  def test_find_optimum(self):
    print ''

    ## Run Optimization for K true
    beta_true = Optim.rho2beta_active(self.rho_true)
    print self.Info_true['msg']
    print '%.5f' % (self.f_true)
    print np2flatstr(beta_true)
    print np2flatstr(self.Ntrue)

    ## Run Optim for K=20
    beta20 = Optim.rho2beta_active(self.rho20)
    print self.Info20['msg']
    print '%.5f' % (self.f20)
    print np2flatstr(beta20[self.grabIDs])
    print np2flatstr(self.N20[self.grabIDs])

    ## Run Optim for K=8
    K8Args = dict(
      alpha = Q20['alpha'],
      gamma = Q20['gamma'],
      nDoc = Q20['nDoc'],
      DocTopicCount = Q20['DocTopicCount'][:, self.grabIDs],
      )
    rho8, omega8, f8, Info = Optim.find_optimum_multiple_tries(
                                      **K8Args)
    beta8 = Optim.rho2beta_active(rho8)
    print Info['msg']
    print '%.5f' % (f8)
    print np2flatstr(beta8)

    assert f8 < self.f20 ## minimization objective!