'''
calcTrueParams.py


'''

import sys
#sys.path.append('~/bnpy/demodata')
#import imp
#MoCap = imp.load_source('MoCap', '/home/will/bnpy/bnpy-dev/demodata/MoCap.py')
import numpy as np
from munkres import Munkres
import scipy.io
import sys




trueK = 12
estK = 12

print sys.argv
filePath = str(sys.argv[1])

X, Z, seqInds = MoCap.get_XZ()
Z -= 1 #Number 0, ..., 11

T = len(Z)



out = scipy.io.loadmat(filePath)
tmp = out['estZ'][0] #for some reason it's a 2-D array..
estZ = []

for seq in xrange(np.shape(tmp)[0]):
  estZ = np.append(estZ, tmp[seq][0])

print np.shape(estZ)
print np.shape(Z)
print T

#Create the cost matrix for munkres algorithm
c = np.zeros((estK, trueK))
for k in xrange(trueK):
  for l in xrange(estK):
    for t in xrange(T):
      if estZ[t] == k and Z[t] != l:
        c[k,l] +=1

#c = np.hstack((c, np.zeros((20,8))))
costs = c.copy()
m = Munkres()
inds = m.compute(c)

daCost = 0
for row, col in inds:
  daCost += costs[row, col]
print 'da cost was: ', daCost

#Permuate the estimated Z's as found by the munkres algorithm
for i in xrange(T):
  estZ[i] = inds[int(estZ[i])][1]

scipy.io.savemat('munkresInfo', {'inds':inds, 'cost':daCost})
  









        



