'''
Unit tests for GaussDistr.py
'''
from bnpy.distr import GaussDistr
import matplotlib.pyplot as plt
import numpy as np

class TestGaussD2(object):
  def setUp(self):
    self.m = np.ones(2)
    self.invSigma = np.eye(2)
    self.distr = GaussDistr(m=self.m, L=self.invSigma)
    
  def test_dimension(self):
    assert self.distr.D == self.invSigma.shape[0]
    
  def test_cholL(self):
    chol = self.distr.cholL()
    assert np.allclose(np.dot(chol, chol.T), self.distr.L)
    
  def test_logdetL(self):
    logdetL = self.distr.logdetL()
    assert np.allclose( np.log(np.linalg.det(self.invSigma)), logdetL)
  
  def test_dist_mahalanobis(self, N=10):
    X = np.random.randn(N, self.distr.D)
    Dist = self.distr.dist_mahalanobis(X)
    invSigma = self.invSigma
    MyDist = np.zeros(N)
    for ii in range(N):
      x = X[ii] - self.m
      MyDist[ii] = np.dot(x.T, np.dot(invSigma, x))
      #if error, we print it out
      print MyDist[ii], Dist[ii]
    assert np.allclose(MyDist, Dist)
  
  def test_sample(self):
      ''' Verifies the function sample produces samples from the appropriate distribution'''
      samples = self.distr.sample(1e+7*self.distr.D)
      
      empirical_mean = np.mean(samples,axis=0)
      print empirical_mean
      
      # allclose tolerance should be scaled with variance, else numSamples needs
      # to be HUGE
      var = 1.0/self.invSigma[0,0]
      
      # test mean
      assert np.allclose(empirical_mean,self.m,atol=1e-3*var)
      print "Empirical mean {} within tolerance of true mean {}".format(empirical_mean,self.m)  
      
      # test cov
                #empirical_cov = np.cov(samples,rowvar=1)
                #print empirical_cov
                #assert np.allclose(empirical_cov.flatten(),np.linalg.inv(self.invSigma).flatten(),atol=1e-1)
                #print "Empirical cov within tolerance of true cov"  
      if(np.size(samples,axis=1)==2):
        #Visual test for covariance. Relaible estiamtes of covarainces require large N and causing numpy's cov function to run out of memory 
        print "True Covaraince {}".format(np.linalg.inv(self.invSigma))
        subSample = np.random.random_integers(0,np.size(samples,axis=0),1e+3)
        plt.figure(1);plt.plot(samples[subSample,0],samples[subSample,1],'y*');
        
    
class TestGaussD1(TestGaussD2):
  def setUp(self):
    self.m = np.ones(1)
    self.invSigma = np.eye(1)
    self.distr = GaussDistr(m=self.m, L=self.invSigma)
    
class TestGaussD2fullCov(TestGaussD2):
    def setUp(self):
        r = np.random.rand();
        self.m = r*np.asarray([-5.,+5.])
        self.invSigma = r*np.linalg.inv(np.asarray([[1.,0.7],[0.7,1.]]))  
        self.distr = GaussDistr(m=self.m, L=self.invSigma)
          
class TestGaussD10(TestGaussD2):
  def setUp(self):
    PRNG = np.random.RandomState(867)
    R = PRNG.rand(10,10)

    self.m = np.ones(10)
    self.invSigma = 1e-4*np.eye(10)
    self.distr = GaussDistr(m=self.m, L=self.invSigma)
    
    
    