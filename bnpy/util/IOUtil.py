import numpy as np

def flatstr2np( xvecstr ):
  return np.asarray( [float(x) for x in xvecstr.split()] )

def np2flatstr( X, fmt="% .6f" ):
  return ' '.join( [fmt%(x) for x in np.asarray(X).flatten() ] )

def np2strList(X, fmt="%.4f", zeroThr=1e-25, zeroSymb=''):
  slist = list()
  for x in np.asarray(X).flatten():
    if np.abs(x) < zeroThr:
      s = zeroSymb
    else:
      s = fmt % (x)
    if np.unique(s[2:]).size == 1 and s.startswith('0.'):
      s = s[1:]
      s = s[:-1] + '1'
      s = '<' + s
    slist.append(s)
  return slist
