import numpy as np
from scipy.special import gammaln, digamma

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import dotATA, dotATB, dotABT
from bnpy.util import as1D, as2D, as3D

from AbstractObsModel import AbstractObsModel


class MultObsModel(AbstractObsModel):

    """ Multinomial data generation model for count vectors.

    Attributes for Prior (Dirichlet)
    --------
    lam : 1D array, size vocab_size
        pseudo-count of observations of each symbol (word) type.

    Attributes for k-th component of EstParams (EM point estimates)
    ---------
    phi[k] : 1D array, size vocab_size
        phi[k] is a vector of positive numbers that sum to one.
        phi[k,v] is probability that vocab type v appears under k.

    Attributes for k-th component of Post (VB parameter)
    ---------
    lam[k] : 1D array, size vocab_size
    """

    def __init__(self, inferType='EM', D=0, vocab_size=0,
                 Data=None, **PriorArgs):
        ''' Initialize bare obsmodel with valid prior hyperparameters.

        Resulting object lacks either EstParams or Post,
        which must be created separately (see init_global_params).
        '''
        if Data is not None:
            self.D = Data.vocab_size
        elif vocab_size > 0:
            self.D = int(vocab_size)
        else:
            self.D = int(D)
        self.K = 0
        self.inferType = inferType
        self.createPrior(Data, **PriorArgs)
        self.Cache = dict()

    def createPrior(self, Data, lam=1.0, min_phi=1e-100, **kwargs):
        ''' Initialize Prior ParamBag attribute.
        '''
        D = self.D
        self.min_phi = min_phi
        self.Prior = ParamBag(K=0, D=D)
        lam = np.asarray(lam, dtype=np.float)
        if lam.ndim == 0:
            lam = lam * np.ones(D)
        assert lam.size == D
        self.Prior.setField('lam', lam, dims=('D'))

    def setupWithAllocModel(self, allocModel):
        ''' Using the allocation model, determine the modeling scenario:
              doc  : multinomial : each atom is D-vector of integer counts
              word : categorical : each atom is a single one-of-D indicator
        '''
        if not isinstance(allocModel, str):
            allocModel = str(type(allocModel))
        aModelName = allocModel.lower()
        if aModelName.count('hdp') or aModelName.count('topic'):
            self.DataAtomType = 'word'
        else:
            self.DataAtomType = 'doc'

    def getTopics(self):
        ''' Retrieve matrix of estimated topic-word probability vectors

        Returns
        --------
        topics : K x vocab_size
                 topics[k,:] is a non-negative vector that sums to one
        '''
        if hasattr(self, 'EstParams'):
            return self.EstParams.phi
        else:
            phi = self.Post.lam / np.sum(self.Post.lam, axis=1)[:, np.newaxis]
            return phi

    def get_name(self):
        return 'Mult'

    def get_info_string(self):
        return 'Multinomial over finite vocabulary.'

    def get_info_string_prior(self):
        msg = 'Dirichlet over finite vocabulary \n'
        if self.D > 2:
            sfx = ' ...'
        else:
            sfx = ''
        S = self.Prior.lam[:2]
        msg += 'lam = %s%s' % (str(S), sfx)
        msg = msg.replace('\n', '\n  ')
        return msg

    def setEstParams(self, obsModel=None, SS=None, LP=None, Data=None,
                     phi=None, topics=None,
                     **kwargs):
        ''' Create EstParams ParamBag with fields phi
        '''
        if topics is not None:
            phi = topics

        self.ClearCache()
        if obsModel is not None:
            self.EstParams = obsModel.EstParams.copy()
            self.K = self.EstParams.K
            return

        if LP is not None and Data is not None:
            SS = self.calcSummaryStats(Data, None, LP)

        if SS is not None:
            self.updateEstParams(SS)
        else:
            self.EstParams = ParamBag(K=phi.shape[0], D=phi.shape[1])
            self.EstParams.setField('phi', phi, dims=('K', 'D'))
        self.K = self.EstParams.K

    def setEstParamsFromPost(self, Post=None, **kwargs):
        ''' Convert from Post (lam) to EstParams (phi),
             each EstParam is set to its posterior mean.
        '''
        if Post is None:
            Post = self.Post
        self.EstParams = ParamBag(K=Post.K, D=Post.D)
        phi = Post.lam / np.sum(Post.lam, axis=1)[:, np.newaxis]
        self.EstParams.setField('phi', phi, dims=('K', 'D'))
        self.K = self.EstParams.K

    def setPostFactors(self, obsModel=None, SS=None, LP=None, Data=None,
                       lam=None, WordCounts=None, **kwargs):
        ''' Set attribute Post to provided values.
        '''
        self.ClearCache()
        if obsModel is not None:
            if hasattr(obsModel, 'Post'):
                self.Post = obsModel.Post.copy()
                self.K = self.Post.K
            else:
                self.setPostFromEstParams(obsModel.EstParams)
            return

        if LP is not None and Data is not None:
            SS = self.calcSummaryStats(Data, None, LP)

        if SS is not None:
            self.updatePost(SS)
        else:
            if WordCounts is not None:
                lam = as2D(WordCounts) + lam
            else:
                lam = as2D(lam)
            K, D = lam.shape
            self.Post = ParamBag(K=K, D=D)
            self.Post.setField('lam', lam, dims=('K', 'D'))
        self.K = self.Post.K

    def setPostFromEstParams(self, EstParams, Data=None, nTotalTokens=0,
                             **kwargs):
        ''' Set attribute Post based on values in EstParams.
        '''
        K = EstParams.K
        D = EstParams.D

        if Data is not None:
            nTotalTokens = Data.word_count.sum()
        if nTotalTokens.ndim == 0:
            nTotalTokens = nTotalTokens / K * np.ones(K)
        assert nTotalTokens.sum() > 0

        WordCounts = EstParams.phi * nTotalTokens[:, np.newaxis]
        lam = WordCounts + self.Prior.lam

        self.Post = ParamBag(K=K, D=D)
        self.Post.setField('lam', lam, dims=('K', 'D'))
        self.K = K

    def calcSummaryStats(self, Data, SS, LP, cslice=(0,None)):
        ''' Calculate summary statistics for given dataset and local parameters

        Returns
        --------
        SS : SuffStatBag object, with K components.
        '''
        if SS is None:
            SS = SuffStatBag(K=LP['resp'].shape[1], D=Data.vocab_size)

        if self.DataAtomType == 'doc':
            # X : 2D sparse matrix, size nDoc x vocab_size
            X = Data.getSparseDocTypeCountMatrix()

            # WordCounts : 2D array, size K x vocab_size
            # obtained by sparse matrix multiply
            # here, '*' operator does this because X is sparse matrix type
            WordCounts = LP['resp'].T * X

        else:
            Resp = LP['resp']  # 2D array, size N x K
            # 2D sparse matrix, size V x N
            X = Data.getSparseTokenTypeCountMatrix()
            if cslice[1] is None:
                WordCounts = (X * Resp).T  # matrix-matrix product
            else:
                start = Data.doc_range[cslice[0]]
                stop = Data.doc_range[cslice[1]]
                X = X[:, start:stop]
                WordCounts = (X * Resp).T  # matrix-matrix product
                
        SS.setField('WordCounts', WordCounts, dims=('K', 'D'))
        SS.setField('SumWordCounts', np.sum(WordCounts, axis=1), dims=('K'))
        return SS

    def forceSSInBounds(self, SS):
        ''' Force count vectors to remain positive
        '''
        np.maximum(SS.WordCounts, 0, out=SS.WordCounts)
        np.maximum(SS.SumWordCounts, 0, out=SS.SumWordCounts)
        if not np.allclose(SS.WordCounts.sum(axis=1), SS.SumWordCounts):
            raise ValueError('Bad Word Counts!')

    def incrementSS(self, SS, k, Data, docID):
        SS.WordCounts[k] += Data.getSparseDocTypeCountMatrix()[docID, :]

    def decrementSS(self, SS, k, Data, docID):
        SS.WordCounts[k] -= Data.getSparseDocTypeCountMatrix()[docID, :]

    def calcLogSoftEvMatrix_FromEstParams(self, Data):
        ''' Compute log soft evidence matrix for Dataset under EstParams.

        Returns
        ---------
        L : 2D array, N x K
        '''
        logphiT = np.log(self.EstParams.phi.T)
        if self.DataAtomType == 'doc':
            X = Data.getSparseDocTypeCountMatrix()
            return X * logphiT
        else:
            return logphiT[Data.word_id, :]

    def updateEstParams_MaxLik(self, SS):
        ''' Update attribute EstParams for all comps given suff stats.

        Update uses the maximum likelihood objective for point estimation.

        Post Condition
        ---------
        Attributes K and EstParams updated in-place.
        '''
        self.ClearCache()
        if not hasattr(self, 'EstParams') or self.EstParams.K != SS.K:
            self.EstParams = ParamBag(K=SS.K, D=SS.D)
        phi = SS.WordCounts / SS.SumWordCounts[:, np.newaxis]
        # prevent entries from reaching exactly 0
        np.maximum(phi, self.min_phi, out=phi)
        self.EstParams.setField('phi', phi, dims=('K', 'D'))

    def updateEstParams_MAP(self, SS):
        ''' Update attribute EstParams for all comps given suff stats.

        Update uses the MAP objective for point estimation.

        Post Condition
        ---------
        Attributes K and EstParams updated in-place.
        '''
        self.ClearCache()
        if not hasattr(self, 'EstParams') or self.EstParams.K != SS.K:
            self.EstParams = ParamBag(K=SS.K, D=SS.D)
        phi = SS.WordCounts + self.Prior.lam - 1
        phi /= phi.sum(axis=1)[:, np.newaxis]
        self.EstParams.setField('phi', phi, dims=('K', 'D'))

    def updatePost(self, SS):
        ''' Update attribute Post for all comps given suff stats.

        Update uses the variational objective.

        Post Condition
        ---------
        Attributes K and Post updated in-place.
        '''
        self.ClearCache()
        if not hasattr(self, 'Post') or self.Post.K != SS.K:
            self.Post = ParamBag(K=SS.K, D=SS.D)

        lam = self.calcPostParams(SS)
        self.Post.setField('lam', lam, dims=('K', 'D'))
        self.K = SS.K

    def calcPostParams(self, SS):
        ''' Calc updated params (lam) for all comps given suff stats

            These params define the common-form of the exponential family
            Dirichlet posterior distribution over parameter vector phi

            Returns
            --------
            lam : 2D array, size K x D
        '''
        Prior = self.Prior
        lam = SS.WordCounts + Prior.lam[np.newaxis, :]
        return lam

    def calcPostParamsForComp(self, SS, kA=None, kB=None):
        ''' Calc params (lam) for specific comp, given suff stats

            These params define the common-form of the exponential family
            Dirichlet posterior distribution over parameter vector phi

            Returns
            --------
            lam : 1D array, size D
        '''
        if kB is None:
            SM = SS.WordCounts[kA]
        else:
            SM = SS.WordCounts[kA] + SS.WordCounts[kB]
        return SM + self.Prior.lam

    def updatePost_stochastic(self, SS, rho):
        ''' Update attribute Post for all comps given suff stats

        Update uses the stochastic variational formula.

        Post Condition
        ---------
        Attributes K and Post updated in-place.
        '''
        assert hasattr(self, 'Post')
        assert self.Post.K == SS.K
        self.ClearCache()

        lam = self.calcPostParams(SS)
        Post = self.Post
        Post.lam[:] = (1 - rho) * Post.lam + rho * lam

    def convertPostToNatural(self):
        ''' Convert current posterior params from common to natural form
        '''
        # Dirichlet common equivalent to natural here.
        pass

    def convertPostToCommon(self):
        ''' Convert current posterior params from natural to common form
        '''
        # Dirichlet common equivalent to natural here.
        pass

    def calcLogSoftEvMatrix_FromPost(self, Data, cslice=(0,None), **kwargs):
        ''' Calculate expected log soft ev matrix under Post.

        Returns
        ------
        L : 2D array, size N x K
        '''
        ElogphiT = self.GetCached('E_logphiT', 'all')  # V x K
        if self.DataAtomType == 'doc':
            X = Data.getSparseDocTypeCountMatrix()  # nDoc x V
            return X * ElogphiT
        else:
            if cslice[1] == None:
                return ElogphiT[Data.word_id, :]
            else:
                start = Data.doc_range[cslice[0]]
                stop = Data.doc_range[cslice[1]]
                wid = Data.word_id[start:stop]
                return ElogphiT[wid, :]

    def calcELBO_Memoized(self, SS, afterMStep=False):
        """ Calculate obsModel's objective using suff stats SS and Post.

        Args
        -------
        SS : bnpy SuffStatBag
        afterMStep : boolean flag
            if 1, elbo calculated assuming M-step just completed

        Returns
        -------
        obsELBO : scalar float
            Equal to E[ log p(x) + log p(phi) - log q(phi)]
        """
        elbo = np.zeros(SS.K)
        Post = self.Post
        Prior = self.Prior
        if not afterMStep:
            Elogphi = self.GetCached('E_logphi', 'all')  # K x V

        for k in xrange(SS.K):
            elbo[k] = c_Diff(Prior.lam, Post.lam[k])
            if not afterMStep:
                elbo[k] += np.inner(SS.WordCounts[k] + Prior.lam - Post.lam[k],
                                    Elogphi[k])
        return np.sum(elbo)

    def logh(self, Data):
        ''' Calculate reference measure for the multinomial distribution

        Returns
        -------
        logh : scalar float, log h(Data) = \sum_{n=1}^N log [ C!/prod_d C_d!]
        '''
        raise NotImplementedError('TODO')

    def getDatasetScale(self, SS, extraSS=None):
        ''' Get number of observed scalars in dataset from suff stats.

        Used for normalizing the ELBO so it has reasonable range.

        Returns
        ---------
        s : scalar positive integer
        '''
        if extraSS is None:
            return SS.SumWordCounts.sum()
        else:
            return SS.SumWordCounts.sum() - extraSS.SumWordCounts.sum()

    def calcHardMergeGap(self, SS, kA, kB):
        ''' Calculate change in ELBO after a hard merge applied to this model

        Returns
        ---------
        gap : scalar real, indicates change in ELBO after merge of kA, kB
        '''
        Prior = self.Prior
        cPrior = c_Func(Prior.lam)

        Post = self.Post
        cA = c_Func(Post.lam[kA])
        cB = c_Func(Post.lam[kB])

        lam = self.calcPostParamsForComp(SS, kA, kB)
        cAB = c_Func(lam)
        return cA + cB - cPrior - cAB

    def calcHardMergeGap_AllPairs(self, SS):
        ''' Calculate change in ELBO for all candidate hard merge pairs

        Returns
        ---------
        Gap : 2D array, size K x K, upper-triangular entries non-zero
              Gap[j,k] : scalar change in ELBO after merge of k into j
        '''
        Prior = self.Prior
        cPrior = c_Func(Prior.lam)

        Post = self.Post
        c = np.zeros(SS.K)
        for k in xrange(SS.K):
            c[k] = c_Func(Post.lam[k])

        Gap = np.zeros((SS.K, SS.K))
        for j in xrange(SS.K):
            for k in xrange(j + 1, SS.K):
                lam = self.calcPostParamsForComp(SS, j, k)
                cjk = c_Func(lam)
                Gap[j, k] = c[j] + c[k] - cPrior - cjk
        return Gap

    def calcHardMergeGap_SpecificPairs(self, SS, PairList):
        ''' Calc change in ELBO for specific list of candidate hard merge pairs

        Returns
        ---------
        Gaps : 1D array, size L
              Gap[j] : scalar change in ELBO after merge of pair in PairList[j]
        '''
        Gaps = np.zeros(len(PairList))
        for ii, (kA, kB) in enumerate(PairList):
            Gaps[ii] = self.calcHardMergeGap(SS, kA, kB)
        return Gaps

    def calcLogMargLikForComp(self, SS, kA, kB=None, **kwargs):
        ''' Calc log marginal likelihood of data assigned to given component

        Args
        -------
        SS : bnpy suff stats object
        kA : integer ID of target component to compute likelihood for
        kB : (optional) integer ID of second component.
             If provided, we merge kA, kB into one component for calculation.
        Returns
        -------
        logM : scalar real
               logM = log p( data assigned to comp kA )
                      computed up to an additive constant
        '''
        return -1 * c_Func(self.calcPostParamsForComp(SS, kA, kB))

    def calcMargLik(self, SS):
        ''' Calc log marginal likelihood combining all comps, given suff stats

            Returns
            --------
            logM : scalar real
                   logM = \sum_{k=1}^K log p( data assigned to comp k | Prior)
        '''
        return self.calcMargLik_CFuncForLoop(SS)

    def calcMargLik_CFuncForLoop(self, SS):
        Prior = self.Prior
        logp = np.zeros(SS.K)
        for k in xrange(SS.K):
            lam = self.calcPostParamsForComp(SS, k)
            logp[k] = c_Diff(Prior.lam, lam)
        return np.sum(logp)

    def _E_logphi(self, k=None):
        if k is None or k == 'prior':
            lam = self.Prior.lam
            Elogphi = digamma(lam) - digamma(np.sum(lam))
        elif k == 'all':
            AMat = self.Post.lam
            Elogphi = digamma(
                AMat) - digamma(np.sum(AMat, axis=1))[:, np.newaxis]
        else:
            Elogphi = digamma(
                self.Post.lam[k]) - digamma(self.Post.lam[k].sum())
        return Elogphi

    def _E_logphiT(self, k=None):
        ''' Calculate transpose of topic-word matrix

            Important to make a copy of the matrix so it is C-contiguous,
            which leads to much much faster matrix operations.

            Returns
            -------
            ElogphiT : 2D array, vocab_size x K
        '''
        ElogphiT = self._E_logphi(k).T.copy()
        return ElogphiT


def c_Func(lam):
    ''' Evaluate cumulant function at given params.

    Returns
    --------
    c : scalar real value of cumulant function at provided args
    '''
    assert lam.ndim == 1
    return gammaln(np.sum(lam)) - np.sum(gammaln(lam))


def c_Diff(lam1, lam2):
    ''' Evaluate difference of cumulant functions c(params1) - c(params2)

    May be more numerically stable than directly using c_Func
    to find the difference.

    Returns
    -------
    diff : scalar real value of the difference in cumulant functions
    '''
    assert lam1.ndim == 1
    assert lam2.ndim == 1
    return c_Func(lam1) - c_Func(lam2)
