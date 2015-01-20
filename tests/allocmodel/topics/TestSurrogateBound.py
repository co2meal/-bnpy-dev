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

LEGENDSIZE=30
TICKSIZE=30
FONTSIZE=40
LINEWIDTH=4

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
              '-', linewidth=LINEWIDTH, label='beta_1=%.2f' % (beta1))

  pylab.xlim([np.min(alphaVals)-0.1, np.max(alphaVals)+0.1])
  pylab.xticks(np.arange(np.max(alphaVals)+1))
  pylab.xlabel("alpha", fontsize=FONTSIZE)

  pylab.ylabel("error", fontsize=FONTSIZE)
  pylab.yticks(np.arange(0, 1.5, 0.5))
  pylab.tick_params(axis='both', which='major', labelsize=TICKSIZE)
  

def plotBoundVsAlph(alphaVals=np.linspace(.001, 3, 1000),
                    beta1=0.5):
  exactVals = cD_exact(alphaVals, beta1)
  boundVals = cD_bound(alphaVals, beta1)

  assert np.all(exactVals >= boundVals)
  pylab.plot( alphaVals, exactVals, 'k-', linewidth=LINEWIDTH)
  pylab.plot( alphaVals, boundVals, 'r--', linewidth=LINEWIDTH)
  pylab.xlabel("alpha", fontsize=FONTSIZE)
  pylab.ylabel("  ", fontsize=FONTSIZE)
  pylab.xlim([np.min(alphaVals)-0.1, np.max(alphaVals)+0.1])
  pylab.ylim([np.min(exactVals)-0.05, np.max(exactVals)+0.05])
  pylab.xticks(np.arange(np.max(alphaVals)+1))

  pylab.legend(['c_D exact', 'c_D surrogate'], fontsize=LEGENDSIZE, loc='lower right')
  pylab.tick_params(axis='both', which='major', labelsize=TICKSIZE)

if __name__ == '__main__':
  from matplotlib import pylab
  pylab.rcParams['ytick.major.pad']='8'

  # Set buffer-space for defining plotable area
  B = 0.18 # big buffer for sides where we will put text labels
  b = 0.02 # small buffer for other sides

  fig1 = pylab.figure(figsize=(10, 6))
  axH = pylab.subplot(111)  
  axH.set_position([B, B, (1-B-b), (1-B-b)])
  plotBoundVsAlph(beta1=0.5)

  fig2 = pylab.figure(figsize=(10, 6))
  axH = pylab.subplot(111)  
  axH.set_position([B, B, (1-B-b), (1-B-b)])
  plotErrorVsAlph(beta1=0.5)
  plotErrorVsAlph(beta1=0.25) 
  plotErrorVsAlph(beta1=0.05)
  plotErrorVsAlph(beta1=0.01)
  pylab.legend(loc='upper left', fontsize=LEGENDSIZE)

  pylab.show(block=False)
  keypress = raw_input('Press y to save, any other key to close >>')
  if keypress.count('y'):
    pylab.figure(1)
    pylab.savefig('SurrogateBound_cDVsAlpha.eps', bbox_inches='tight', format='eps')
    pylab.figure(2)
    pylab.savefig('SurrogateBound_ErrorVsAlpha.eps', bbox_inches='tight')

  pylab.show(block=True)
  
