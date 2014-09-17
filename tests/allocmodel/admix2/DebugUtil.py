import numpy as np

def assert_allclose(a, b, aname='', bname='', 
                          atol=1e-8, rtol=1e-8, Ktop=5, fmt='% .6f'):
  if len(aname) > 0:
    print '------------------------- %s' % (aname.split('_')[0])
    printVectors(a, b, aname, bname)

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

def printVectors(a, b=0, aname='', bname='', fmt='%9.6f', Kmax=10, start=0):
  if len(a) > 2*Kmax:
    print 'FIRST %d' % (Kmax)
    printVectors(a[:Kmax], b[:Kmax], aname, bname, fmt, Kmax)
    print 'LAST %d' % (Kmax)
    printVectors(a[-Kmax:], b[-Kmax:], aname, bname, fmt, Kmax)

  else:
    print ' %16s %s' % (aname, np2flatstr(a, fmt, Kmax))
    if bname is not None:
      print ' %16s %s' % (bname, np2flatstr(b, fmt, Kmax))

def np2flatstr(xvec, fmt='%9.3f', Kmax=10, start=0):
  return ' '.join( [fmt % (x) for x in xvec[start:start+Kmax]])
