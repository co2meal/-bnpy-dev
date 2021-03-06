import copy
import numpy as np
from ParamBag import ParamBag


class SuffStatBag(object):

    """ Container object for additive sufficient statistics in bnpy.

    Uses ParamBag as internal representation.

    Tracks three possible sets of parameters, each with own ParamBag:
    * sufficient statistics fields
    * (optional) precomputed ELBO terms
    * (optional) precomputed terms for potential merges
    """

    def __init__(self, K=0, **kwargs):
        self.kwargs = kwargs
        self._Fields = ParamBag(K=K, **kwargs)

    def setUIDs(self, uIDs):
        if len(uIDs) != self.K:
            raise ValueError('Bad UIDs')
        self.uIDs = np.asarray(uIDs, dtype=np.int32)

    def getCountVec(self):
        ''' Return vector of counts for each active topic/component
        '''
        if 'N' in self._Fields._FieldDims:
            return self.N
        elif 'SumWordCounts' in self._Fields._FieldDims:
            return self.SumWordCounts
        raise ValueError('Counts not available')

    def copy(self, includeELBOTerms=True, includeMergeTerms=True):
        if not includeELBOTerms:
            E = self.removeELBOTerms()
        if not includeMergeTerms:
            M = self.removeMergeTerms()
        copySS = copy.deepcopy(self)
        if not includeELBOTerms:
            self.restoreELBOTerms(E)
        if not includeMergeTerms:
            self.restoreMergeTerms(M)
        return copySS

    def setField(self, key, value, dims=None):
        self._Fields.setField(key, value, dims=dims)

    def setAllFieldsToZeroAndRemoveNonELBOTerms(self):
        self._Fields.setAllFieldsToZero()
        self.setELBOFieldsToZero()
        self.removeMergeTerms()
        self.removeSelectionTerms()

    def setELBOFieldsToZero(self):
        if self.hasELBOTerms():
            self._ELBOTerms.setAllFieldsToZero()

    def setMergeFieldsToZero(self):
        if self.hasMergeTerms():
            self._MergeTerms.setAllFieldsToZero()

        if hasattr(self, 'mPairIDs'):
            delattr(self, 'mPairIDs')

    def reorderComps(self, order):
        ''' Rearrange internal order of components
        '''
        assert self.K == order.size
        self._Fields.reorderComps(order)
        if self.hasELBOTerms():
            self._ELBOTerms.reorderComps(order)
        if self.hasMergeTerms():
            self._MergeTerms.reorderComps(order)
        if self.hasSelectionTerms():
            self._SelectTerms.reorderComps(order)
        if hasattr(self, 'uIDs'):
            self.uIDs = self.uIDs[order]
        if hasattr(self, 'mPairIDs'):
            del self.mPairIDs

    def removeField(self, key):
        return self._Fields.removeField(key)

    def removeELBOandMergeTerms(self):
        E = self.removeELBOTerms()
        M = self.removeMergeTerms()
        return E, M

    def restoreELBOandMergeTerms(self, E, M):
        self.restoreELBOTerms(E)
        self.restoreMergeTerms(M)

    def removeELBOTerms(self):
        if not self.hasELBOTerms():
            return None
        _ELBOTerms = self._ELBOTerms
        del self._ELBOTerms
        return _ELBOTerms

    def removeMergeTerms(self):
        if hasattr(self, 'mPairIDs'):
            del self.mPairIDs
        if not self.hasMergeTerms():
            return None
        MergeTerms = self._MergeTerms
        del self._MergeTerms
        return MergeTerms

    def restoreELBOTerms(self, ELBOTerms):
        if ELBOTerms is not None:
            self._ELBOTerms = ELBOTerms

    def restoreMergeTerms(self, MergeTerms):
        if MergeTerms is not None:
            self._MergeTerms = MergeTerms

    def removeSelectionTerms(self):
        if not self.hasSelectionTerms():
            return None
        STerms = self._SelectTerms
        del self._SelectTerms
        return STerms

    def restoreSelectionTerms(self, STerms):
        if STerms is not None:
            self._SelectTerms = STerms

    def hasAmpFactor(self):
        return hasattr(self, 'ampF')

    def applyAmpFactor(self, ampF):
        self.ampF = ampF
        for key in self._Fields._FieldDims:
            arr = getattr(self._Fields, key)
            if arr.ndim == 0:
                # Edge case: in-place updates don't work with
                # de-referenced 0-d arrays
                setattr(self._Fields, key, arr * ampF)
            else:
                arr *= ampF

    def hasELBOTerms(self):
        return hasattr(self, '_ELBOTerms')

    def hasELBOTerm(self, key):
        if not hasattr(self, '_ELBOTerms'):
            return False
        return hasattr(self._ELBOTerms, key)

    def getELBOTerm(self, key):
        return getattr(self._ELBOTerms, key)

    def setELBOTerm(self, key, value, dims=None):
        if not hasattr(self, '_ELBOTerms'):
            self._ELBOTerms = ParamBag(K=self.K, **self.kwargs)
        self._ELBOTerms.setField(key, value, dims=dims)

    def hasMergeTerms(self):
        return hasattr(self, '_MergeTerms')

    def hasMergeTerm(self, key):
        if not hasattr(self, '_MergeTerms'):
            return False
        return hasattr(self._MergeTerms, key)

    def getMergeTerm(self, key):
        return getattr(self._MergeTerms, key)

    def setMergeTerm(self, key, value, dims=None):
        if not hasattr(self, '_MergeTerms'):
            self._MergeTerms = ParamBag(K=self.K, **self.kwargs)
        self._MergeTerms.setField(key, value, dims=dims)

    def hasSelectionTerm(self, key):
        if not hasattr(self, '_SelectTerms'):
            return False
        return hasattr(self._SelectTerms, key)

    def hasSelectionTerms(self):
        return hasattr(self, '_SelectTerms')

    def getSelectionTerm(self, key):
        return getattr(self._SelectTerms, key)

    def setSelectionTerm(self, key, value, dims=None):
        if not hasattr(self, '_SelectTerms'):
            self._SelectTerms = ParamBag(K=self.K)
        self._SelectTerms.setField(key, value, dims=dims)

    def multiMergeComps(self, kdel, alph):
        ''' Blend comp kdel into all remaining comps k
        '''
        if self.K <= 1:
            raise ValueError('Must have at least 2 components to merge.')
        for key, dims in self._Fields._FieldDims.items():
            if dims is not None and dims != ():
                arr = getattr(self._Fields, key)
                for k in xrange(self.K):
                    if k == kdel:
                        continue
                    arr[k] += alph[k] * arr[kdel]

        if self.hasELBOTerms() and self.hasMergeTerms():
            Ndel = getattr(self._Fields, 'N')[kdel]
            Halph = -1 * np.inner(alph, np.log(alph + 1e-100))
            Hplus = self.getMergeTerm('ElogqZ')[kdel]
            gap = Ndel * Halph - Hplus
            for k in xrange(self.K):
                if k == kdel:
                    continue
                arr = getattr(self._ELBOTerms, 'ElogqZ')
                arr[k] -= gap * alph[k]

        self._Fields.removeComp(kdel)
        if self.hasELBOTerms():
            self._ELBOTerms.removeComp(kdel)
        if self.hasMergeTerms():
            self._MergeTerms.removeComp(kdel)
        if self.hasSelectionTerms():
            self._SelectTerms.removeComp(kdel)

    def mergeComps(self, kA, kB):
        ''' Merge components kA, kB into a single component
        '''
        if self.K <= 1:
            raise ValueError('Must have at least 2 components to merge.')
        if kB == kA:
            raise ValueError('Distinct component ids required.')
        for key, dims in self._Fields._FieldDims.items():
            if dims is not None and dims != ():
                arr = getattr(self._Fields, key)
                if self.hasMergeTerm(key) and dims == ('K'):
                    # some special terms need to be precomputed
                    arr[kA] = getattr(self._MergeTerms, key)[kA, kB]
                elif dims == ('K', 'K'):
                    # special case for HMM transition matrix
                    arr[kA] += arr[kB]
                    arr[:, kA] += arr[:, kB]
                elif dims[0] == 'K':
                    # applies to vast majority of all fields
                    arr[kA] += arr[kB]
                elif len(dims) > 1 and dims[1] == 'K':
                    arr[:, kA] += arr[:, kB]

        if hasattr(self, 'mPairIDs'):
            # Find the right row that corresponds to input kA, kB
            mPairIDs = self.mPairIDs
            rowID = np.flatnonzero(np.logical_and(kA == mPairIDs[:, 0],
                                                  kB == mPairIDs[:, 1]))
            if not rowID.size == 1:
                raise ValueError(
                    'Bad search for correct mPairID.\n' + str(rowID))
            rowID = rowID[0]

        if self.hasELBOTerms():
            for key, dims in self._ELBOTerms._FieldDims.items():
                if not self.hasMergeTerm(key):
                    continue

                arr = getattr(self._ELBOTerms, key)
                mArr = getattr(self._MergeTerms, key)
                mdims = self._MergeTerms._FieldDims[key]

                if mdims[0] == 'M':
                    mArr = mArr[rowID]
                    if mArr.ndim == 2 and mArr.shape[0] == 2:
                        arr[kA, :] = mArr[0]
                        arr[:, kA] = mArr[1]
                    elif mArr.ndim <= 1 and mArr.size == 1:
                        arr[kA] = mArr
                    else:
                        raise NotImplementedError('TODO')

                elif dims[0] == 'K':
                    if mArr.ndim == 2:
                        arr[kA] = mArr[kA, kB]
                    else:
                        arr[kA] += mArr[kB]

        if self.hasMergeTerms():
            for key, dims in self._MergeTerms._FieldDims.items():
                mArr = getattr(self._MergeTerms, key)
                if dims == ('K', 'K'):
                    mArr[kA, kA + 1:] = np.nan
                    mArr[:kA, kA] = np.nan
                elif dims == ('K'):
                    mArr[kA] = np.nan
                elif dims[0] == 'M' and key == 'Htable':
                    if len(dims) == 3 and dims[-1] == 'K':
                        mArr[:, :, kA] = 0

        if self.hasSelectionTerms():
            for key, dims in self._SelectTerms._FieldDims.items():
                mArr = getattr(self._SelectTerms, key)
                if dims == ('K', 'K'):
                    ab = mArr[kB, kB] + 2 * mArr[kA, kB] + mArr[kA, kA]
                    mArr[kA, :] += mArr[kB, :]
                    mArr[:, kA] += mArr[:, kB]
                    mArr[kA, kA] = ab
                elif dims == ('K'):
                    mArr[kA] = mArr[kA] + mArr[kB]

        if hasattr(self, 'mPairIDs'):
            # Remove any other pairs associated with kA or kB
            keepRowIDs = ((mPairIDs[:, 0] != kA) *
                          (mPairIDs[:, 1] != kA) *
                          (mPairIDs[:, 0] != kB) *
                          (mPairIDs[:, 1] != kB))

            keepRowIDs = np.flatnonzero(keepRowIDs)
            M = len(keepRowIDs)
            self.M = M
            self._MergeTerms.M = M

            # Remove any other pairs related to kA, kB
            for key, dims in self._MergeTerms._FieldDims.items():
                mArr = getattr(self._MergeTerms, key)
                if dims[0] == 'M':
                    mArr = mArr[keepRowIDs]
                    self._MergeTerms.setField(key, mArr, dims=dims)

            # Reindex mPairIDs
            mPairIDs = mPairIDs[keepRowIDs]
            mPairIDs[mPairIDs[:, 0] > kB, 0] -= 1
            mPairIDs[mPairIDs[:, 1] > kB, 1] -= 1
            self.mPairIDs = mPairIDs

        if hasattr(self, 'uIDs'):
            self.uIDs = np.delete(self.uIDs, kB)
        self._Fields.removeComp(kB)
        if self.hasELBOTerms():
            self._ELBOTerms.removeComp(kB)
        if self.hasMergeTerms():
            self._MergeTerms.removeComp(kB)
        if self.hasSelectionTerms():
            self._SelectTerms.removeComp(kB)

    def insertComps(self, SS):
        self._Fields.insertComps(SS)
        if hasattr(self, '_ELBOTerms'):
            self._ELBOTerms.insertEmptyComps(SS.K)
        if hasattr(self, '_MergeTerms'):
            self._MergeTerms.insertEmptyComps(SS.K)
        if hasattr(self, '_SelectTerms'):
            self._SelectTerms.insertEmptyComps(SS.K)

    def insertEmptyComps(self, Kextra):
        self._Fields.insertEmptyComps(Kextra)
        if hasattr(self, '_ELBOTerms'):
            self._ELBOTerms.insertEmptyComps(Kextra)
        if hasattr(self, '_MergeTerms'):
            self._MergeTerms.insertEmptyComps(Kextra)
        if hasattr(self, '_SelectTerms'):
            self._SelectTerms.insertEmptyComps(Kextra)

    def removeComp(self, k):
        if hasattr(self, 'uIDs'):
            self.uIDs = np.delete(self.uIDs, k)
        self._Fields.removeComp(k)
        if hasattr(self, '_ELBOTerms'):
            self._ELBOTerms.removeComp(k)
        if hasattr(self, '_MergeTerms'):
            self._MergeTerms.removeComp(k)
        if hasattr(self, '_SelectTerms'):
            self._SelectTerms.removeComp(k)

    def getComp(self, k, doCollapseK1=True):
        SS = SuffStatBag(K=1, D=self.D)
        SS._Fields = self._Fields.getComp(k, doCollapseK1=doCollapseK1)
        return SS

    def __add__(self, PB):
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        SSsum = SuffStatBag(K=self.K, D=self.D)
        SSsum._Fields = self._Fields + PB._Fields

        if hasattr(self, '_ELBOTerms') and hasattr(PB, '_ELBOTerms'):
            SSsum._ELBOTerms = self._ELBOTerms + PB._ELBOTerms
        elif hasattr(PB, '_ELBOTerms'):
            SSsum._ELBOTerms = PB._ELBOTerms.copy()

        if hasattr(self, '_MergeTerms') and hasattr(PB, '_MergeTerms'):
            SSsum._MergeTerms = self._MergeTerms + PB._MergeTerms
        elif hasattr(PB, '_MergeTerms'):
            SSsum._MergeTerms = PB._MergeTerms.copy()

        if hasattr(self, '_SelectTerms') and hasattr(PB, '_SelectTerms'):
            SSsum._SelectTerms = self._SelectTerms + PB._SelectTerms

        if not hasattr(self, 'mPairIDs') and hasattr(PB, 'mPairIDs'):
            SSsum.mPairIDs = PB.mPairIDs
        return SSsum

    def __iadd__(self, PB):
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        self._Fields += PB._Fields

        if hasattr(self, '_ELBOTerms') and hasattr(PB, '_ELBOTerms'):
            self._ELBOTerms += PB._ELBOTerms
        elif hasattr(PB, '_ELBOTerms'):
            self._ELBOTerms = PB._ELBOTerms.copy()

        if hasattr(self, '_MergeTerms') and hasattr(PB, '_MergeTerms'):
            self._MergeTerms += PB._MergeTerms
        elif hasattr(PB, '_MergeTerms'):
            self._MergeTerms = PB._MergeTerms.copy()

        if hasattr(self, '_SelectTerms') and hasattr(PB, '_SelectTerms'):
            self._SelectTerms += PB._SelectTerms

        if not hasattr(self, 'mPairIDs') and hasattr(PB, 'mPairIDs'):
            self.mPairIDs = PB.mPairIDs
        return self

    def __sub__(self, PB):
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        SSsum = SuffStatBag(K=self.K, D=self.D)
        SSsum._Fields = self._Fields - PB._Fields
        if hasattr(self, '_ELBOTerms') and hasattr(PB, '_ELBOTerms'):
            SSsum._ELBOTerms = self._ELBOTerms - PB._ELBOTerms
        if hasattr(self, '_MergeTerms') and hasattr(PB, '_MergeTerms'):
            SSsum._MergeTerms = self._MergeTerms - PB._MergeTerms
        return SSsum

    def __isub__(self, PB):
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        self._Fields -= PB._Fields
        if hasattr(self, '_ELBOTerms') and hasattr(PB, '_ELBOTerms'):
            self._ELBOTerms -= PB._ELBOTerms
        if hasattr(self, '_MergeTerms') and hasattr(PB, '_MergeTerms'):
            self._MergeTerms -= PB._MergeTerms
        return self

    def __getattr__(self, key):
        _Fields = object.__getattribute__(self, "_Fields")
        _dict = object.__getattribute__(self, "__dict__")
        if key == "_Fields":
            return _Fields
        elif hasattr(_Fields, key):
            return getattr(_Fields, key)
        elif key == '__deepcopy__':  # workaround to allow copying
            return None
        elif key in _dict:
            return _dict[key]
        # Field named 'key' doesnt exist.
        errmsg = "'SuffStatBag' object has no attribute '%s'" % (key)
        raise AttributeError(errmsg)

    '''
    def __getstate__(self):
        F = self._Fields
        E = None
        M = None
        S = None
        if self.hasELBOTerms():
            E = self._ELBOTerms
        if self.hasMergeTerms():
            M = self._MergeTerms
        if self.hasSelectionTerms():
            S = self._SelectTerms
        return F, E, M, S

    def __setstate__(self, state):
        F, E, M, S = state
        self._Fields = F
        if E is not None:
            self._ELBOTerms = E
        if M is not None:
            self._MergeTerms = M
        if S is not None:
            self._SelectTerms = S
        self.K = F.K
    '''
