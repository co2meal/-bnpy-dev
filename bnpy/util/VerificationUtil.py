'''
VerificationUtil.py

Verification utilities, for checking whether numerical variables are "equal".
'''
import numpy as np

def isEvenlyDivisibleFloat(a, b, margin=1e-6):
  ''' Returns true/false for whether a is evenly divisible by b 
        within a (small) numerical tolerance
      Examples
      --------
      >>> isEvenlyDivisibleFloat( 1.5, 0.5)
      True
      >>> isEvenlyDivisibleFloat( 1.0, 1./3)
      True
  '''
  cexact = np.asarray(a)/float(b)
  cround = np.round(cexact)
  return abs(cexact - cround) < margin
  
def assert_allclose(a, b, atol=1e-8, rtol=0):
    """ Verify two arrays a,b are numerically indistinguishable.
    """
    isOK = np.allclose(a, b, atol=atol, rtol=rtol)
    if not OK:
       msg = np2flatstr(a)
       msg += "\n"
       msg += np2flatstr(b)
       print msg
    assert isOK
    
