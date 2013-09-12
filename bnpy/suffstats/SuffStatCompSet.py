'''
SuffStatCompSet.py

Sufficient statistics for any bnpy mixture model

Attributes
--------
K : integer number of components
Nvec : 1D-array of expected counts for all components
comps : list of SuffStat objects, one per component
'''

import numpy as np

from SuffStat import SuffStat

class SuffStatCompSet(object):

  def __init__( self, Nvec=0):
    self.Nvec = np.asarray(Nvec)
    self.K = self.Nvec.size
    self.comp = None
  
  def fillComps( self, ConstructorFunc, **kwargs):
    self.comp = [None for k in range(self.K)]
    for kk in xrange(self.K):
      curDict = dict([ (k,v[kk]) for k,v in kwargs.items()])
      self.comp[kk] = ConstructorFunc( N=self.Nvec[kk], **curDict)

  def getNvec(self):
    return self.Nvec
  
  def getNtotal(self):
    return np.sum(Nvec)
    
  def setAmpFactor(self,ampF):
    self.ampF = ampF
  
  def hasAmpFactor(self):
    return hasattr(self,'ampF')
  
  def applyAmpFactor(self):
    for comp in self.comp:
      comp.applyAmpFactor(self.ampF)  
     
  def addPrecompEntropy(self, Hvec):
    self.Hvec = Hvec
  
  def hasPrecompEntropy(self):
    return hasattr(self, 'Hvec')
      
  def addComp( self, SScomp, posID=None):
    if posID is None:
      posID = self.K
    self.comps.insert( posID, SScomp )
    self.K = len(self.comp)
    
  def addEmptyComp(self):
    SScomp = self.comp[0].MakeEmpty()
    self.comps.append( SScomp)
    self.K = len(self.comp)
      
  def removeComp(self, posID):
    if posID < 0 or posID >= self.K:
      raise ValueError('Invalid position. 0 <= posID < K')
    del self.comp[posID]
    self.K -= 1
    assert len(self.comp)==self.K
    
  def addManyComps(self, SScompset):
    self.comp.extend(SScompset.comp)
    self.K = len(self.comp)