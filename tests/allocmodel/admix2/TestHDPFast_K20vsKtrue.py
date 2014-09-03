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
    self.alpha = 0.5 # prefers Kmany when gamma=10
    #self.alpha = 0.99 # prefers Ktrue when gamma=10
    
    self.gamma = 10
    #self.gamma = 100

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
      alpha = self.alpha,
      gamma = self.gamma,
      nDoc = Qtrue['nDoc'],
      DocTopicCount = Qtrue['DocTopicCount'],
      )
    rhot, omegat, ft, Info = Optim.find_optimum_multiple_tries(**TrueArgs)
    self.rho_true = rhot
    self.omega_true = omegat
    self.f_true = ft
    self.Info_true = Info

    K20Args = dict(
      alpha = self.alpha,
      gamma = self.gamma,
      nDoc = Q20['nDoc'],
      DocTopicCount = Q20['DocTopicCount'],
      )
    rho20, omega20, f20, Info = Optim.find_optimum_multiple_tries(
                                      **K20Args)
    self.rho20=rho20
    self.omega20=omega20
    self.f20 = f20
    self.Info20 = Info

    K8Args = dict(
      alpha = self.alpha,
      gamma = self.gamma,
      nDoc = Q20['nDoc'],
      DocTopicCount = Q20['DocTopicCount'][:, self.grabIDs],
      )
    rho8, omega8, f8, Info8 = Optim.find_optimum_multiple_tries(**K8Args)
    self.rho8 = rho8
    self.omega8 = omega8
    self.f8 = f8
    self.Info8 = Info8

  def test_calc_evidence(self):
    ''' Compare Ktrue and K20 models
    '''
    print ''

    ## Make Ktrue model
    mtrue = HDPFast('VB', dict(alpha=self.alpha, gamma=self.gamma))
    mtrue.rho = self.rho_true
    mtrue.omega = self.omega_true
    mtrue.K = mtrue.rho.size
    SS = SuffStatBag(K=mtrue.K, D=2, A=5)
    SS.setField('nDoc', 5.0, dims=None)
    SS.setField('DocTopicCount', Qtrue['DocTopicCount'], dims=('A', 'K'))
    SS.setELBOTerm('ElogqZ', np.zeros(8), dims='K')
    ELBOtrue = mtrue.calc_evidence(None, SS, None)

    ## Make K20 model
    m20 = HDPFast('VB', dict(alpha=self.alpha, gamma=self.gamma))
    m20.rho = self.rho20
    m20.omega = self.omega20
    m20.K = m20.rho.size
    SS20 = SuffStatBag(K=m20.K, D=2, A=5)
    SS20.setField('nDoc', 5.0, dims=None)
    SS20.setField('DocTopicCount', Q20['DocTopicCount'], dims=('A', 'K'))
    SS20.setELBOTerm('ElogqZ', np.zeros(20), dims='K')
    ELBO20 = m20.calc_evidence(None, SS20, None)

    print '------------- Compare  ELBO gap computation'
    print '%.6f  | ELBO Ktrue' % (ELBOtrue)
    print '%.6f  | ELBO K20' % (ELBO20)

    if ELBO20 > ELBOtrue:
      print 'WINNER: K20'
    else:
      print 'WINNER: Ktrue'

    print ''
    print '------------- Verify ELBO gap computation'
    print '%.6f  |  GAP Ktrue - K20 using calc_evidence()' \
                     % (ELBOtrue - ELBO20) 

    fgap = -1 * (self.f_true - self.f20) * SS.nDoc
    fgap += (8 - 20) * Optim.c_Beta(1, self.gamma) 
    print '%.6f  |  GAP Ktrue - K20 using find_optimum() obj func' % (fgap) 

    print '------------- TOTAL GAP (including D K log alpha)'
    agap = SS.nDoc * (SS.K - SS20.K) * np.log(self.alpha)
    print '%.6f | GAP including missing alpha term' % (fgap + agap)


  def test_inputs_equal_in_nonzero_columns(self):
    ''' Verify that Ntrue and 8 columns of N20 are exactly the same.
    '''
    print ''

    N8 = self.N20[self.grabIDs]
    print np2flatstr(self.Ntrue)
    print np2flatstr(N8)
    assert np.allclose(self.Ntrue, N8)


  def test_find_optimum(self):
    ''' Verify that top8 solution optimum is exactly equal to Ktrue optimum
    '''
    print ''

    ## Ktrue solution
    print '-------------------- Ktrue'
    beta_true = Optim.rho2beta_active(self.rho_true)
    print '%.5f' % (self.f_true)
    print np2flatstr(beta_true)

    ## K20 solution
    print '-------------------- K20'
    beta20 = Optim.rho2beta_active(self.rho20)
    print '%.5f' % (self.f20)
    print np2flatstr(beta20[self.grabIDs])

    ## Top 8 Columns of K8 solution
    print '-------------------- Ktop8'
    beta8 = Optim.rho2beta_active(self.rho8)
    print '%.5f' % (self.f8)
    print np2flatstr(beta8)

    ## Verify top8 soln equal to true 
    assert np.allclose(self.f_true, self.f8)

    ## Verify that top 8 columns solution better than all 20
    ## Here, better means less than due to minimization objective!
    assert self.f8 < self.f20 