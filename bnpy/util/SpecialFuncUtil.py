from collections import defaultdict
import numpy as np
from scipy.special import gammaln, digamma

LOGPI = np.log(np.pi)
LOGTWO = np.log(2.)
LOGTWOPI = np.log( 2.*np.pi )
EPS = 10*np.finfo(float).eps

def closeAtMSigFigs(A, B, M=10, tol=5):
  ''' Returns true/false for whether A and B are numerically "close"
          aka roughly equal at M significant figures

      Only makes sense for numbers on scale of abs. value 1.0 or larger.      
      Log evidences will definitely always be at this scale.

      Examples
      --------
      >>> closeAtMSigFigs(1234, 1000, M=1)  # margin is 500 
      True
      >>> closeAtMSigFigs(1234, 1000, M=2)  # margin is 50 
      False
      >>> closeAtMSigFigs(1034, 1000, M=2)  # margin is 50 
      True
      >>> closeAtMSigFigs(1005, 1000, M=3)  # margin is 5 
      True
      >>> closeAtMSigFigs(44.5, 49.5, M=1) # margin is 5 
      True
      >>> closeAtMSigFigs(44.5, 49.501, M=1)
      False
      >>> closeAtMSigFigs(44.499, 49.5, M=1) 
      False
  '''
  A = float(A)
  B = float(B)
  # Enforce abs(A) >= abs(B)
  if abs(A) < abs(B):
    tmp = A
    A = B
    B = tmp
  assert abs(A) >= abs(B)

  # Find the scale that A (the larger of the two) possesses
  #  A ~= 10 ** (P10)
  P10 = int(np.floor(np.log10(abs(A))))

  # Compare the difference between A and B
  #   to the allowed margin THR
  diff = abs(A - B)
  if P10 >= 0:
    THR = tol * 10.0**(P10 - M)
    THR = (1 + 1e-11) * THR 
    # make THR just a little bigger to avoid issues where 2.0 and 1.95
    # aren't equal at 0.05 margin due to rounding errors
    return np.sign(A) == np.sign(B) and diff <= THR
  else:
    THR = tol * 10.0**(-M)
    THR = (1 + 1e-11) * THR
    return diff <= THR


MVgCache = defaultdict( lambda: dict())
def MVgammaln(x, D):
  ''' Compute log of the D-dimensional multivariate Gamma func. for input x
          
      Notes: Caching gives big speedup!
      -------
       caching : 208 sec for 5 iters of CGS on K=50, D=2 problem with N=10000
      no cache : 300 sec
  '''
  try:
    return MVgCache[D][x]
  except KeyError:
    result = gammaln(x+ 0.5*(1 - np.arange(1,D+1)) ).sum() + 0.25*D*(D-1)*LOGPI
    MVgCache[D][x] = result
  return result
  
def MVdigamma(x, D):
  ''' Compute the first-derivative of the log of the D-dim. Gamma function
  '''
  return digamma(x + 0.5 * (1 - np.arange(1,D+1))).sum()

def logsumexp(logA, axis=None):
  ''' Efficiently compute log(sum(exp(...))) for input matrix "logA"
      Computation is both vectorized and numerically stable.
  '''
  logA = np.asarray( logA )
  logAmax = logA.max( axis=axis )

  if axis is None:
    logA = logA - logAmax
  elif axis==1:
    logA = logA - logAmax[:,np.newaxis]
  elif axis==0:
    logA = logA - logAmax[np.newaxis,:]
  assert np.allclose( logA.max(), 0.0 )
  logA = np.log( np.sum( np.exp(logA), axis=axis )  )
  return logA + logAmax
