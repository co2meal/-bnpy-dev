'''
SuffStatBag.py
'''
import numpy as np
from ParamBag import ParamBag

class SuffStatBag(object):
  def __init__(self, K=0, D=0):
    self._Fields = ParamBag(K=K, D=D)
    self._ELBOTerms = ParamBag(K=K)
    self._MergeTerms = ParamBag(K=K)

  def setField(self, key, value, dims=None):
    self._Fields.setField(key, value, dims=dims)

  # ======================================================= ELBO terms
  def getELBOTerm(self, key):
    return getattr(self._ELBOTerms, key)

  def setELBOTerm(self, key, value, dims=None):
    self._ELBOTerms.setField(key, value, dims=dims)

  # ======================================================= ELBO merge terms
  def getMergeTerm(self, key):
    return getattr(self._MergeTerms, key)

  def setMergeTerm(self, key, value, dims=None):
    self._MergeTerms.setField(key, value, dims=dims)


  def hasAmpFactor(self):
    return 'ampF' in self.__dict__
        
  def applyAmpFactor(self, ampF):
    self.ampF = ampF
    for key in self._Fields._FieldDims:
      arr = getattr(self._Fields, key)
      arr *= ampF
      

  # ======================================================= Merge comps
  def mergeComps(self, kA, kB):
    ''' Merge components kA, kB into a single component
    '''
    if self.K <= 1:
      raise ValueError('Must have at least 2 components to merge.')
    if kB == kA:
      raise ValueError('Distinct component ids required.')
    SA = self._Fields.getComp(kA)
    SB = self._Fields.getComp(kB)
    self._Fields.setComp(kA, SA + SB)

    for key, dims in self._ELBOTerms._FieldDims.items():
      if key in self._MergeTerms._FieldDims and (dims == ('K') or dims == 'K'):
        arr = getattr(self._ELBOTerms, key)
        mArr = getattr(self._MergeTerms, key)
        print arr, mArr
        arr[kA] = mArr[kA,kB]

    for key, dims in self._MergeTerms._FieldDims.items():
      if dims == ('K','K'):
        mArr = getattr(self._MergeTerms, key)
        mArr[kA,kA+1:] = np.nan
        mArr[:kA,kA] = np.nan

    self._Fields.removeComp(kB)
    self._ELBOTerms.removeComp(kB)
    self._MergeTerms.removeComp(kB)

  # ======================================================= Insert comps
  def insertComps(self, SS):
    self._Fields.insertComps(SS)
    self._ELBOTerms.insertEmptyComps(SS.K)
    self._MergeTerms.insertEmptyComps(SS.K)

  def insertEmptyComps(self, Kextra):
    self._Fields.insertEmptyComps(Kextra)
    self._ELBOTerms.insertEmptyComps(Kextra)
    self._MergeTerms.insertEmptyComps(Kextra)

  # ======================================================= Remove comp
  def removeComp(self, k):
    self._Fields.removeComp(k)
    self._ELBOTerms.removeComp(k)
    self._MergeTerms.removeComp(k)

  # ======================================================= Get comp
  def getComp(self, k):
    return self._Fields.getComp(k)

  # ======================================================= Override add
  def __add__(self, PB):
    if self.K != PB.K or self.D != PB.D:
      raise ValueError('Dimension mismatch')
    SSsum = SuffStatBag(K=self.K, D=self.D)
    SSsum._Fields = self._Fields + PB._Fields
    SSsum._ELBOTerms = self._ELBOTerms + PB._ELBOTerms
    SSsum._MergeTerms = self._MergeTerms + PB._MergeTerms
    return SSsum

  def __iadd__(self, PB):
    if self.K != PB.K or self.D != PB.D:
      raise ValueError('Dimension mismatch')
    self._Fields += PB._Fields
    self._ELBOTerms += PB._ELBOTerms
    self._MergeTerms += PB._MergeTerms
    return self

  # ======================================================= Override subtract
  def __sub__(self, PB):
    if self.K != PB.K or self.D != PB.D:
      raise ValueError('Dimension mismatch')
    SSsum = SuffStatBag(K=self.K, D=self.D)
    SSsum._Fields = self._Fields - PB._Fields
    SSsum._ELBOTerms = self._ELBOTerms - PB._ELBOTerms
    SSsum._MergeTerms = self._MergeTerms - PB._MergeTerms
    return SSsum

  def __isub__(self, PB):
    if self.K != PB.K or self.D != PB.D:
      raise ValueError('Dimension mismatch')
    self._Fields -= PB._Fields
    self._ELBOTerms -= PB._ELBOTerms
    self._MergeTerms -= PB._MergeTerms
    return self

  # ======================================================= Override getattr
  def __getattr__(self, key):
    if key in self.__dict__:
      return self.__dict__[key]
    elif hasattr(self._Fields, key):
      return getattr(self._Fields,key)
    raise KeyError('Unknown field %s' % (key))