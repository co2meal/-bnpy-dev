'''  Timing results on my macbook and dept machine wolfgang,
         with matrix X = np.random.rand( 1e6, 64)
      A | fblas.dgemm(1.0, X, X, trans_a=True)
      B | fblas.dgemm(1.0, X.T, X.T, trans_b=True)
      C | np.dot(X.T,X)
                   C         A        B      
      macbook      1.46 s    1.20 s    0.69 s
      wolfgang     1.67 s    1.45 s    0.58 s
'''
import numpy as np
import scipy.linalg.fblas

def dotATB( A, B):
  if A.shape[1] > B.shape[1]:
    return scipy.linalg.fblas.dgemm(1.0, A, B, trans_a=True)
  else:
    return np.dot( A.T, B)

def dotABT( A, B):
  if B.shape[0] > A.shape[0]:
    return scipy.linalg.fblas.dgemm(1.0, A, B, trans_b=True)
  else:
    return np.dot( A, B.T)
    
def dotATA( A ):
  return scipy.linalg.fblas.dgemm(1.0, A.T, A.T, trans_b=True)
