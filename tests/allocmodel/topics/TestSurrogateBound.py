'''
TestSurrogateBound.py

Verify the lower bound on the Dirichlet cumulant function.

Usage
---------
To plot bound for manual study
$ python TestSurrogateBound.py

To verify automatically
$ nosetests -v TestSurrogateBound
'''

import numpy as np
from scipy.special import gammaln
import unittest
import bnpy

TICKSIZE=20
FONTSIZE=25


def cD_exact(alphaVals, beta1):
  return gammaln(alphaVals) \
            - gammaln(alphaVals * beta1) \
            - gammaln(alphaVals * (1-beta1)) 

def cD_bound(alphaVals, beta1):
  return np.log(alphaVals) \
             + np.log(beta1) \
             + np.log((1-beta1)) 


class TestSurrogateBound(unittest.TestCase):

  def shortDescription(self):
    return None

  def test_is_lower_bound(self):
    ''' Verify that cD_bound does in fact provide a lower bound of cD_exact
    '''
    for beta1 in np.linspace(1e-2, 0.5, 10):
      alphaVals = np.linspace(.00001, 10, 1000)
      exactVals = cD_exact(alphaVals, beta1)
      boundVals = cD_bound(alphaVals, beta1)        
      assert np.all(exactVals >= boundVals)



def plotErrorVsAlph(alphaVals=np.linspace(.001, 3, 1000),
                    beta1=0.5):
  exactVals = cD_exact(alphaVals, beta1)
  boundVals = cD_bound(alphaVals, beta1)
  assert np.all(exactVals >= boundVals)
  pylab.plot(alphaVals, exactVals - boundVals, 
              '-', linewidth=2, label='beta_1=%.2f' % (beta1))

  pylab.xlabel("alpha", fontsize=FONTSIZE)
  pylab.ylabel("error", fontsize=FONTSIZE)
  pylab.tick_params(axis='both', which='major', labelsize=TICKSIZE)
  

def plotBoundVsAlph(alphaVals=np.linspace(.001, 3, 1000),
                    beta1=0.5):
  exactVals = cD_exact(alphaVals, beta1)
  boundVals = cD_bound(alphaVals, beta1)

  assert np.all(exactVals >= boundVals)
  pylab.plot( alphaVals, exactVals, 'k-', linewidth=2)
  pylab.plot( alphaVals, boundVals, 'r--', linewidth=2)
  pylab.xlabel("alpha", fontsize=FONTSIZE)

  pylab.legend(['c_D exact', 'c_D surrogate'], fontsize=FONTSIZE, loc='lower right')
  pylab.tick_params(axis='both', which='major', labelsize=TICKSIZE)

if __name__ == '__main__':
  from matplotlib import pylab
  # Set buffer-space for defining plotable area
  B = 0.15 # big buffer for sides where we will put text labels
  b = 0.05 # small buffer for other sides

  fig1 = pylab.figure(figsize=(8, 6))
  axH = pylab.subplot(111)  
  axH.set_position([B, B, (1-B-b), (1-B-b)])
  plotBoundVsAlph(beta1=0.5)
  
  fig2 = pylab.figure(figsize=(8, 6))
  axH = pylab.subplot(111)  
  axH.set_position([B, B, (1-B-b), (1-B-b)])
  plotErrorVsAlph(beta1=0.5)
  plotErrorVsAlph(beta1=0.25) 
  plotErrorVsAlph(beta1=0.05)
  plotErrorVsAlph(beta1=0.01)
  pylab.legend(loc='upper left', fontsize=FONTSIZE)

  #pylab.figure(fig1)
  #pylab.savefig('SurrogateBound_cDVsAlpha.eps', bbox_inches='tight')
  #pylab.figure(fig2)
  #pylab.savefig('SurrogateBound_ErrorVsAlpha.eps', bbox_inches='tight')

  pylab.show(block=True)
  
