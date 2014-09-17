import sys
import numpy as np
from nose.plugins.skip import Skip, SkipTest
from scipy.optimize import approx_fprime
import warnings
import unittest

from bnpy.allocmodel.admix2.HDPSB import gtsum
import bnpy.allocmodel.admix2.LocalOptimizerDir as Optim

import BarsK10V900

def MakeSingleDocProblem(K=0, alpha=0, seed=0, nWordsPerDoc=100):
  Data = BarsK10V900.get_data(seed=seed, nDocTotal=1, nWordsPerDoc=nWordsPerDoc)
  Lik = np.log(Data.TrueParams['topics']).T[Data.word_id].copy()
  Lik = Lik[:, :K]
  Lik = np.exp(Lik - np.max(Lik, axis=1)[:,np.newaxis])

  Ebeta = np.hstack([(np.ones(K) - 0.01) / K, 0.01])
  assert np.allclose(Ebeta.sum(), 1.0)
  return dict(wc_d=Data.word_count, Lik_d=Lik,
              alphaEbeta=alpha * Ebeta, Data=Data)

########################################################### TestK1
###########################################################
""" when K=1, we have a simple analytic solution
theta[0] = Nd + alphaEbeta[0]
theta[1] = alphaEbeta[1]
"""

class TestK1(unittest.TestCase):
  def shortDescription(self):
    return None

  def testHasSaneOutput__objFunc_constrained(self):
    ''' Verify objective value and gradient vector have correct type and size
    '''
    print ''
    for K in [1]:
      for alpha in [0.1, 0.99]:
        for seed in [333, 777, 888]:
          print '==================  K %5d | alpha %.2f' % (K, alpha)
          for approx_grad in [1, 0]:
            PRNG = np.random.RandomState(seed)
            ProbDict = MakeSingleDocProblem(K=K, seed=seed)
            theta = 5 * PRNG.rand(K+1)
            if approx_grad:
              f = Optim.objFunc_constrained(theta, approx_grad=1,
                                                 **ProbDict)
              g = np.ones(K+1)
            else:
              f, g = Optim.objFunc_constrained(theta, approx_grad=0,
                                                 **ProbDict)

            print ' f= ', f
            assert type(f) == np.float64
            assert g.ndim == 1
            assert g.size == K+1            
            assert np.isfinite(f)
            assert np.all(np.isfinite(g))

  def testGradientZeroAtOptimum__objFunc_constrained(self):
    ''' Verify computed gradient at optimum is indistinguishable from zero
    '''
    print ''
    for K in [1]:
      for alpha in [0.1, 0.95]:
        print '==================  K %5d | alpha %.2f' % (K, alpha)

        ProbDict = MakeSingleDocProblem(K=K, seed=0, alpha=alpha)
        theta1 = ProbDict['wc_d'].sum() + ProbDict['alphaEbeta'][0]
        thetaRem = ProbDict['alphaEbeta'][-1]
        theta = np.hstack([theta1, thetaRem])
        
        ## Exact gradient
        _, g = Optim.objFunc_constrained(theta,
                                          approx_grad=0,
                                          **ProbDict)

        ## Numerical gradient
        objFunc = lambda x: Optim.objFunc_constrained(x,
                                                          approx_grad=1,
                                                          **ProbDict)
        epsvec = 1e-8*np.ones(K+1)
        gapprox = approx_fprime(theta, objFunc, epsvec)    
        
        assert_allclose(g, gapprox, 'g', 'gapprox', atol=0.01, rtol=1e-4)
        assert_allclose(g, np.zeros(g.size), 'g', 'zero', atol=0.001)

  def testGradientExactAndApproxAgree__objFunc_constrained(self):
    ''' Verify computed gradient similar for exact and approx methods
    '''
    print ''
    for K in [1]:
      for alpha in [0.1, 0.95]:
        print '==================  K %5d | alpha %.2f' % (K, alpha)

        for seed in [111, 222, 333]:

          PRNG = np.random.RandomState(seed)
          ProbDict = MakeSingleDocProblem(K=K, seed=seed, alpha=alpha)
          theta = PRNG.rand(K+1)

          ## Exact gradient
          _, g = Optim.objFunc_constrained(theta,
                                           approx_grad=0,
                                           **ProbDict)

          ## Numerical gradient
          objFunc = lambda x: Optim.objFunc_constrained(x,
                                                        approx_grad=1,
                                                        **ProbDict)
          epsvec = 1e-8 * np.ones(K+1)
          gapprox = approx_fprime(theta, objFunc, epsvec)    

          assert_allclose(g, gapprox, 'g', 'gapprox', atol=1e-4, rtol=1e-4)

  def testRecoverIdeal__find_optimum(self):
    ''' Verify find_optimum estimates the known ideal solution for K=1
    '''
    print ''
    for K in [1]:
      for alpha in [0.1, 0.95]:
        print '==================  K %5d | alpha %.2f' % (K, alpha)

        for seed in [111, 222, 333]:

          PRNG = np.random.RandomState(seed)
          ProbDict = MakeSingleDocProblem(K=K, seed=seed, alpha=alpha)
          inittheta = PRNG.rand(K+1)
          thetaopt = np.hstack([
                      ProbDict['wc_d'].sum() + ProbDict['alphaEbeta'][0], 
                      ProbDict['alphaEbeta'][1]  ])
          fopt = Optim.objFunc_constrained(thetaopt, approx_grad=1,
                                             **ProbDict)

          thetaest, fest, Info = Optim.find_optimum(inittheta=inittheta,
                                                    approx_grad=0,
                                                    **ProbDict)
          print Info['task']

          thetah0t, fh0t, Info = Optim.find_optimum(inittheta=thetaopt,
                                                    approx_grad=0,
                                                    **ProbDict)
          print Info['task']

          print 'f_est %.8f' % (fest)
          print 'f_opt %.8f' % (fopt)
          print 'f_hot %.8f' % (fh0t)
          print ProbDict['wc_d'].size
          assert_allclose(thetah0t, thetaopt, 'thetahot', 'thetaopt', atol=0.01)

          assert_allclose(thetaest, thetaopt, 'thetaest', 'thetaopt', 
                                               atol=0.01, rtol=0.001)

########################################################### Test with Many Docs
###########################################################
class TestManyK(unittest.TestCase):
  def shortDescription(self):
    return None

  def testHasSaneOutput__objFunc_constrained(self):
    ''' Verify objective value and gradient vector have correct type and size
    '''
    print ''
    for K in [2, 7, 10]: # max value of K is 10
      for seed in [33, 77, 888]:
        for alpha in [0.1, 0.9]:
          PRNG = np.random.RandomState(seed)
          ProbDict = MakeSingleDocProblem(K=K, seed=seed, alpha=alpha)
          theta = PRNG.rand(K+1)
          for approx_grad in [0, 1]:
            if approx_grad:
              f = Optim.objFunc_constrained(theta, approx_grad=1,
                                               **ProbDict)
              g = np.ones(K+1)
              fapprox = f
            else:
              f, g = Optim.objFunc_constrained(theta, approx_grad=0,
                                                 **ProbDict)
              fexact = f
            assert type(f) == np.float64
            assert g.ndim == 1
            assert g.size == K + 1            
            assert np.isfinite(f)
            assert np.all(np.isfinite(g))
          assert np.allclose(fexact, fapprox)

  def testGradientExactAndApproxAgree__objFunc_constrained(self):
    ''' Verify computed gradient similar for exact and approx methods
    '''
    print ''
    for K in [2, 4, 10]:
      for alpha in [0.1, 0.95]:
        print '================== K %d | alpha %.2f' \
                  % (K, alpha)

        for seed in [111, 222, 333]:
          PRNG = np.random.RandomState(seed)
          theta = 10 * PRNG.rand(K+1)    
          ProbDict = MakeSingleDocProblem(K=K, seed=seed, alpha=alpha)

          ## Exact gradient
          f, g = Optim.objFunc_constrained(theta, approx_grad=0,
                                                 **ProbDict)

          ## Approx gradient
          objFunc = lambda x: Optim.objFunc_constrained(x, approx_grad=1,
                                                               **ProbDict)
          epsvec = 1e-8 * np.ones(K+1)
          gapprox = approx_fprime(theta, objFunc, epsvec)    


          assert_allclose(g, gapprox, 'g', 'gapprox', atol=1e-4, rtol=1e-4)


  def testRecoverSameAsCoordAscent(self):
    ''' Give same input to find_optimum and LocalStep. Verify output same.
    '''
    from bnpy.allocmodel.admix2 import LocalUtil
    from scipy.special import digamma

    print ''
    for K in [2, 5, 10]:
      for alpha in [0.95]:
        print '================== K %d | alpha %.2f' \
                  % (K, alpha)

        for seed in [111, 222]:
          PRNG = np.random.RandomState(seed)
          inittheta = 10 * PRNG.rand(K+1)    
          ProbDict = MakeSingleDocProblem(K=K, seed=seed, alpha=alpha)

          def aModelFunc(DocTopicCount_d):
            theta = ProbDict['alphaEbeta'].copy()
            theta[:-1] += DocTopicCount_d
            ElogPi = digamma(theta[:-1]) - digamma(theta.sum())
            return ElogPi

          DocTopicCount_d = np.zeros(K)
          Prior_d = np.ones(K)
          sumR_d = np.zeros(ProbDict['wc_d'].size)
          DocTopicCount_d, Prior_d, sumR_d = LocalUtil.calcDocTopicCountForDoc(
                            0, 
                            aModelFunc,
                            DocTopicCount_d, ProbDict['Lik_d'],
                            Prior_d, sumR_d, 
                            wc_d=ProbDict['wc_d'],
                            nCoordAscentItersLP=500,
                            convThrLP=0.0001)
          thetaCA = ProbDict['alphaEbeta'].copy()
          thetaCA[:-1] += DocTopicCount_d

          thetahot, fhot, Info = Optim.find_optimum(inittheta=thetaCA,
                                                    approx_grad=0,
                                                    **ProbDict)
          print Info['task']

          assert_allclose(thetaCA, thetahot, 'thetaCA', 'thetahot',
                           atol=0.01, rtol=0.001)
          print ''



  def testShowLocalOptima(self):
    ''' Show different local optima that exist on Bars Data
    '''
    from bnpy.viz.BarsViz import plotExampleBarsDocs, pylab
  
    for K in [10]:
      for alpha in [0.95]:
        print '================== K %d | alpha %.2f' \
                  % (K, alpha)
        ## good seed = 11, 4545
        ProbDict = MakeSingleDocProblem(K=K, seed=888, alpha=alpha)

        InitPiList = list()
        initscoreList = list()
        EPiList = list()
        scoreList = list()
        for seed in xrange(16):
          PRNG = np.random.RandomState(seed)
          inittheta = 10 * PRNG.rand(K+1)
          initscore = Optim.objFunc_constrained(inittheta, approx_grad=1, 
                                                **ProbDict)

          InitPiList.append( inittheta[:-1] / np.sum(inittheta) )
          initscoreList.append(initscore)

          thetaest, fest, Info = Optim.find_optimum(inittheta=inittheta,
                                                    approx_grad=0,
                                                    **ProbDict)
          EPi = thetaest[:K] / np.sum(thetaest)
          EPiList.append(EPi)
          scoreList.append(fest)

        ## Sort matrix so similar rows are clustered together
        scores = np.hstack(scoreList)
        sortIDs = np.argsort(-1 * scores)
        EPiMat =  np.vstack(EPiList)[sortIDs]
        scores = scores[sortIDs]
        initscores = np.hstack(initscoreList)[sortIDs]
        InitPiMat = np.vstack(InitPiList)[sortIDs]

        Data = ProbDict['Data']
        plotExampleBarsDocs(Data, [0])
        showPiFromManyRuns(pylab, EPiMat, scores, Data)
        #showPiFromManyRuns(pylab, InitPiMat, initscores, Data)
        pylab.show(block=True)


def showPiFromManyRuns(pylab, EPiMat, scores, Data):
  sqrtW = np.sqrt(Data.vocab_size)
  K = EPiMat.shape[1]
  nOnTopicWords = sqrtW / float(0.5*K)
  ticks = np.linspace(0, nOnTopicWords*(K/2-1), K/2) \
          + 0.5*nOnTopicWords

  pylab.subplots(nrows=4, ncols=4)
  for row in xrange(16):
    EPi = EPiMat[row]
    Topic = np.dot(EPi, Data.TrueParams['topics'])
    SqTopicIm = np.reshape(Topic, (sqrtW, sqrtW))      

    figH = pylab.subplot(4, 4, row+1)
    pylab.imshow( SqTopicIm, vmin=0, vmax=5.0/Data.vocab_size)

    pylab.yticks(ticks, np2flatstr(EPi[:K/2], fmt='%.2f').split(' '))
    pylab.xticks(ticks, np2flatstr(EPi[K/2:], fmt='%.2f').split(' '))
    pylab.title( '%.3f' % (scores[row]), fontsize=10 )
    ax = pylab.gca()
    ax.tick_params(labelsize=8)
  pylab.tight_layout()

def assert_allclose(a, b, aname='', bname='', 
                          atol=1e-8, rtol=1e-8, Ktop=5, fmt='% .6f'):
  if len(aname) > 0:
    print '------------------------- %s' % (aname.split('_')[0])
    printVectors(aname, a, bname, b)
  isOK = np.allclose(a, b, atol=atol, rtol=rtol)
  if not isOK:
    print 'VIOLATION DETECTED!'
    print 'args are not equal (within tolerance)'
                
    absDiff = np.abs(a - b)
    tolDiff = (atol + rtol * np.abs(b)) - absDiff
    worstIDs = np.argsort(tolDiff)
    print 'Top %d worst mismatches' % (Ktop)
    print np2flatstr( a[worstIDs[:Ktop]], fmt=fmt)
    print np2flatstr( b[worstIDs[:Ktop]], fmt=fmt)
  assert isOK

def printVectors(aname, a, bname=None, b=np.zeros(2), fmt='%9.6f', Kmax=10):
  if len(a) > Kmax + 2:
    print 'FIRST %d' % (Kmax)
    printVectors(aname, a[:Kmax], bname, b[:Kmax], fmt, Kmax)
    print 'LAST %d' % (Kmax)
    printVectors(aname, a[-Kmax:], bname, b[-Kmax:], fmt, Kmax)

  else:
    print ' %10s %s' % (aname, np2flatstr(a, fmt, Kmax))
    if bname is not None:
      print ' %10s %s' % (bname, np2flatstr(b, fmt, Kmax))

def np2flatstr(xvec, fmt='%9.3f', Kmax=10):
  return ' '.join( [fmt % (x) for x in xvec[:Kmax]])
