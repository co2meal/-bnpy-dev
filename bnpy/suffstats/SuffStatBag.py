import copy
import numpy as np
from ParamBag import ParamBag


class SuffStatBag(object):

    """ Container object for additive sufficient statistics in bnpy.

    Uses ParamBag as internal representation.

    Attributes
    ----------
    * K : int
        number of components
    * uids : 1D array, size K
        unique ids of the components
    * _Fields : ParamBag
        track relevant values
    * _ELBOTerms : optional ParamBag, default does not exist
        precomputed ELBO terms
    * _MergeTerms : optional ParamBag, default does not exist
        precomputed terms for candidate merges
    * xSS : optional dict of SuffStatBags, default does not exist
        keys of the dict are uids
    """

    def __init__(self, K=0, uids=None, **kwargs):
        '''

        Post Condition
        ---------------
        Creates an empty SuffStatBag object,
        with valid values of uids and K.
        '''
        self._Fields = ParamBag(K=K, **kwargs)
        if uids is None:
            self.uids = np.arange(K, dtype=np.int32)
        else:
            self.uids = np.asarray(uids, dtype=np.int32).copy()
        self._kwargs = kwargs

    def setUIDs(self, uids):
        ''' Set the unique comp ids to new values.

        Post Condition
        --------------
        Attribute uids updated if provided array-like was valid.
        '''
        if len(uids) != self.K:
            emsg = 'Bad uids. Expected length %d, got %d.' % (
                self.K, len(uids))
            raise ValueError(emsg)
        self.uids = np.asarray(uids, dtype=np.int32)

    def uid2k(self, uid):
        ''' Indentify the position index of provided uid.
        '''
        k = np.flatnonzero(self.uids == uid)
        if k.size < 1:
            raise ValueError('Cannot find uid %d' % (uid))
        elif k.size > 1:
            raise ValueError('Badness. Multiple copies of uid %d' % (uid))
        return k[0]

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
        ''' Set named field to provided array-like value.

        Thin wrapper around ParamBag's setField method.
        '''
        self._Fields.setField(key, value, dims=dims)

    def setAllFieldsToZeroAndRemoveNonELBOTerms(self):
        ''' Fill all arrays in _Fields to zeroes and remove merge terms.
        '''
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
        if hasattr(self, 'mUIDPairs'):
            delattr(self, 'mUIDPairs')

    def reorderComps(self, order=None, uids=None,
                     fieldsToIgnore=['sumLogPiRemVec']):
        ''' Rearrange internal order of components.
        '''
        if uids is not None:
            uids = np.asarray(uids, dtype=np.int32)
            order = np.zeros_like(uids)
            for pos in range(uids.size):
                order[pos] = self.uid2k(uids[pos])
        else:
            order = np.asarray(order, dtype=np.int32)
        assert self.K == order.size
        assert hasattr(self, 'uids')
        self.uids = self.uids[order]
        assert self.uids.size == order.size
        self._Fields.reorderComps(order, fieldsToIgnore)
        if hasattr(self, 'mUIDPairs'):
            del self.mUIDPairs
        if self.hasELBOTerms():
            self._ELBOTerms.reorderComps(order)
        if self.hasMergeTerms():
            self._MergeTerms.reorderComps(order)
        if self.hasSelectionTerms():
            self._SelectTerms.reorderComps(order)

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
        if hasattr(self, 'mUIDPairs'):
            del self.mUIDPairs
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
            self._ELBOTerms = ParamBag(K=self.K, **self._kwargs)
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
            self._MergeTerms = ParamBag(K=self.K, **self._kwargs)
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


    def insertComps(self, SS):
        self._Fields.insertComps(SS)
        self.uids = np.hstack([self.uids, SS.uids])
        if hasattr(self, '_ELBOTerms'):
            if SS.hasELBOTerms():
                self._ELBOTerms.insertComps(SS._ELBOTerms)
            else:
                self._ELBOTerms.insertEmptyComps(SS.K)
        if hasattr(self, '_MergeTerms'):
            self._MergeTerms.insertEmptyComps(SS.K)
        if hasattr(self, '_SelectTerms'):
            self._SelectTerms.insertEmptyComps(SS.K)

    def insertEmptyComps(self, Kextra, newuids=None):
        if newuids is None:
            uidstart = self.uids.max()+1
            newuids = np.arange(uidstart, uidstart+Kextra)
        self._Fields.insertEmptyComps(Kextra)
        self.uids = np.hstack([self.uids, newuids])
        if hasattr(self, '_ELBOTerms'):
            self._ELBOTerms.insertEmptyComps(Kextra)
        if hasattr(self, '_MergeTerms'):
            self._MergeTerms.insertEmptyComps(Kextra)
        if hasattr(self, '_SelectTerms'):
            self._SelectTerms.insertEmptyComps(Kextra)

    def removeComp(self, k=None, uid=None):
        ''' Remove any value associated with index k from tracked values.
        '''
        if k is None and uid is not None:
            k = self.uid2k(uid)
        self.uids = np.delete(self.uids, k)
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

    def findMergePairByUID(self, uidA, uidB):
        ''' Find which currently tracked merge pair contains desired uids.

        Returns
        -------
        rowID : int
            index of tracked merge quantities related to specific uid pair. 
        '''
        assert hasattr(self, 'mUIDPairs')
        rowID = np.flatnonzero(
            np.logical_and(uidA == self.mUIDPairs[:, 0],
                           uidB == self.mUIDPairs[:, 1]))
        if not rowID.size == 1:
            raise ValueError(
                'Bad search for correct merge UID pair.\n' + str(rowID))
        rowID = rowID[0]

    def mergeComps(self, kA=None, kB=None, uidA=None, uidB=None):
        ''' Merge components kA, kB into a single component, in-place.

        Post Condition
        --------------
        This SuffStatBag will have K-1 states, one less than before the call.
        All fields related to [kA] will have values combined with [kB].
        All fields related to [kB] will then be removed/deleted.
        '''
        if self.K <= 1:
            raise ValueError('Must have at least 2 components to merge.')

        if kA is None or kB is None:
            kA = np.flatnonzero(self.uids == uidA)
            kB = np.flatnonzero(self.uids == uidB)
        else:
            uidA = self.uids[kA]
            uidB = self.uids[kB]
        assert kA is not None
        assert kB is not None

        if kB == kA:
            raise ValueError('Distinct component ids required.')

        # Find the right row that corresponds to input kA, kB
        if hasattr(self, 'mUIDPairs'):
            rowID = self.findMergePairByUID(uidA, uidB)
        else:
            rowID = None

        # Fill entry [kA] of each field with correct value.
        self._mergeFieldsAtIndexKA(kA, kB, rowID)
        # Fill entry [kA] of each elboterm with correct value.
        self._mergeELBOTermsAtIndexKA(kA, kB, rowID)

        self._setMergeTermsAtIndexKAToNaN(kA, kB, rowID)

        self._mergeSelectionTermsAtIndexKA(kA, kB, rowID)

        self._discardAnyTrackedPairsThatOverlapWithAorB(uidA, uidB)
        self.uids = np.delete(self.uids, kB)
        assert uidA in self.uids
        assert uidB not in self.uids
        
        # Finally, remove dimension kB from all fields
        self._Fields.removeComp(kB)
        if self.hasELBOTerms():
            self._ELBOTerms.removeComp(kB)
        if self.hasMergeTerms():
            self._MergeTerms.removeComp(kB)
        if self.hasSelectionTerms():
            self._SelectTerms.removeComp(kB)

    def _discardAnyTrackedPairsThatOverlapWithAorB(self, uidA, uidB):
        ''' Update to discard remaining pairs that overlap uidA/uidB.

        Post Condition
        --------------
        Attributes mUIDPairs and _MergeTerms dont have any more info
        about other pairs (uidj,uidk) where where uidA or uidB are involved.
        '''
        if hasattr(self, 'mUIDPairs'):
            mUIDPairs = self.mUIDPairs
            # Remove any other pairs associated with kA or kB
            keepRowIDs = ((mUIDPairs[:, 0] != uidA) *
                          (mUIDPairs[:, 1] != uidA) *
                          (mUIDPairs[:, 0] != uidB) *
                          (mUIDPairs[:, 1] != uidB))
            keepRowIDs = np.flatnonzero(keepRowIDs)
            self.mUIDPairs = mUIDPairs[keepRowIDs]
            self.M = len(keepRowIDs)
            self._MergeTerms.M = self.M
            # Remove any other pairs related to kA, kB
            for key, dims in self._MergeTerms._FieldDims.items():
                mArr = getattr(self._MergeTerms, key)
                if dims[0] == 'M':
                    mArr = mArr[keepRowIDs]
                    self._MergeTerms.setField(key, mArr, dims=dims)

    def _mergeFieldsAtIndexKA(self, kA, kB, rowID):
        ''' For each field, rewrite values for comp kA to merge kA, kB.

        Post Condition
        --------------
        Every key, arr pair in _Fields will have size K, as before.
        The array will have entries related to component kA overwritten.
        '''
        for key, dims in self._Fields._FieldDims.items():
            if dims is not None and dims != ():
                # Get numpy array object for field named by key
                arr = getattr(self._Fields, key)
                assert arr.ndim >= 1

                # Now edit this array in place
                if self.hasMergeTerm(key) and dims == ('K'):
                    # Use precomputed term stored under _MergeTerms
                    arr[kA] = getattr(self._MergeTerms, key)[kA, kB]
                elif dims == ('K', 'K'):
                    # Special logic for HMM transition matrix
                    arr[kA] += arr[kB]
                    arr[:, kA] += arr[:, kB]
                elif dims[0] == 'K':
                    # applies to vast majority of all fields
                    arr[kA] += arr[kB]
                elif len(dims) > 1 and dims[1] == 'K':
                    arr[:, kA] += arr[:, kB]

    def _mergeELBOTermsAtIndexKA(self, kA, kB, rowID):
        ''' For each ELBOterm, rewrite values for comp kA to merge kA, kB.

        Post Condition
        --------------
        Every key, arr pair in _ELBOTerms will have size K, as before.
        The array will have entries related to component kA overwritten.
        '''
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

    def _setMergeTermsAtIndexKAToNaN(self, kA, kB, rowID):
        ''' Make terms tracked for kA incompatible for future merges.
        '''        
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

    def _mergeSelectionTermsAtIndexKA(self, kA, kB, rowID):
        ''' Update terms at index kA.
        '''
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

    def replaceCompWithExpansion(self, uid=0, xSS=None, 
            keysToSetNonExtraZero=['sumLogPiRemVec']):
        ''' Replace existing component with expanded set of statistics.

        Post Condition
        --------------
        Values associated with uid are removed.
        All entries of provided xSS are added last in index order.
        '''
        if not np.intersect1d(xSS.uids, self.uids).size == 0:
            raise ValueError("Cannot expand with same uids.")

        for key in self._Fields._FieldDims:
            if key in keysToSetNonExtraZero:
                arr = getattr(self._Fields, key)
                arr.fill(0)
        self.insertComps(xSS)
        self.removeComp(uid=uid)


    def __add__(self, PB):
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        if not np.allclose(self.uids, PB.uids):
            raise ValueError('Cannot combine stats for differing uids.')
        SSsum = SuffStatBag(K=self.K, D=self.D, uids=self.uids)
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
        if not np.allclose(self.uids, PB.uids):
            raise ValueError('Cannot combine stats for differing uids.')
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
        if not np.allclose(self.uids, PB.uids):
            raise ValueError('Cannot combine stats for differing uids.')
        SSsum = SuffStatBag(K=self.K, D=self.D, uids=self.uids)
        SSsum._Fields = self._Fields - PB._Fields
        if hasattr(self, '_ELBOTerms') and hasattr(PB, '_ELBOTerms'):
            SSsum._ELBOTerms = self._ELBOTerms - PB._ELBOTerms
        if hasattr(self, '_MergeTerms') and hasattr(PB, '_MergeTerms'):
            SSsum._MergeTerms = self._MergeTerms - PB._MergeTerms
        return SSsum

    def __isub__(self, PB):
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        if not np.allclose(self.uids, PB.uids):
            raise ValueError('Cannot combine stats for differing uids.')
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

    """
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
    """