

def runKMeans_BregmanDiv_existing(
        X, K, obsModel,
        W=None,
        Niter=100,
        seed=0,
        smoothFracInit=1.0,
        smoothFrac=0,
        logFunc=None,
        eps=1e-10,
        setOneToPriorMean=0,
        distexp=1.0,
        init='plusplus',
        **kwargs):
    ''' Run hard clustering algorithm to add K new clusters to existing model.

    Returns
    -------
    Z : 1D array, size N
        Contains assignments to Korig + K possible clusters
        if Niter == 0, unassigned data items have value Z[n] = -1 

    Mu : 2D array, size (Korig + K) x D
        Includes original Korig clusters and K new clusters

    Lscores : 1D array, size Niter
    '''
    Korig = obsModel.K
    chosenZ, Mu, _, _ = initKMeans_BregmanDiv_exist(
        X, K, obsModel,
        W=W,
        seed=seed,
        smoothFrac=smoothFracInit,
        distexp=distexp,
        setOneToPriorMean=setOneToPriorMean)
    # Make sure we update K to reflect the returned value.
    # initKMeans_BregmanDiv will return fewer than K clusters
    # in some edge cases, like when data matrix X has duplicate rows
    # and specified K is larger than the number of unique rows.
    K = len(Mu) - Korig
    assert K > 0
    assert Niter >= 0
    if Niter == 0:
        Z = -1 * np.ones(X.shape[0])
        if chosenZ[0] == -1:
            Z[chosenZ[1:]] = Korig + np.arange(chosenZ.size - 1)
        else:
            Z[chosenZ] = Korig + np.arange(chosenZ.size)
    Lscores = list()
    prevN = np.zeros(K)
    for riter in xrange(Niter):
        Div = obsModel.calcSmoothedBregDiv(
            X=X, Mu=Mu, W=W,
            includeOnlyFastTerms=True, 
            smoothFrac=smoothFrac, eps=eps)
        Z = np.argmin(Div, axis=1)
        Ldata = Div.min(axis=1).sum()
        Lprior = obsModel.calcBregDivFromPrior(
            Mu=Mu, smoothFrac=smoothFrac).sum()
        Lscore = Ldata + Lprior
        Lscores.append(Lscore)
        # Verify objective is monotonically increasing
        try:
            # Test allows small positive increases that are
            # numerically indistinguishable from zero. Don't care about these.
            assert np.all(np.diff(Lscores) <= 1e-5)
        except AssertionError:
            msg = 'In the kmeans update loop of FromScratchBregman.py'
            msg += 'Lscores not monotonically decreasing...'
            if logFunc:
                logFunc(msg)
            else:
                print msg
            assert np.all(np.diff(Lscores) <= 1e-5)

        N = np.zeros(K)
        for k in xrange(K):
            if W is None:
                W_k = None
                N[k] = np.sum(Z==k)
            else:
                W_k = W[Z==k]
                N[k] = np.sum(W_k)
            if N[k] > 0:
                Mu[k] = obsModel.calcSmoothedMu(X[Z==k], W_k)
            else:
                Mu[k] = obsModel.calcSmoothedMu(X=None)
        if logFunc:
            logFunc("iter %d: Lscore %.3e" % (riter, Lscore))
            if W is None:
                 str_sum_w = ' '.join(['%7.0f' % (x) for x in N])
            else:
                 assert np.allclose(N.sum(), W.sum())
                 str_sum_w = ' '.join(['%7.2f' % (x) for x in N])
            str_sum_w = split_str_into_fixed_width_lines(str_sum_w, tostr=True)
            logFunc(str_sum_w)
        if np.max(np.abs(N - prevN)) == 0:
            break
        prevN[:] = N

    uniqueZ = np.unique(Z)
    if Niter > 0:
        # In case a cluster was pushed to zero
        if uniqueZ.size < len(Mu):
            Mu = [Mu[k] for k in uniqueZ]
    else:
        # Without full pass through dataset, many items not assigned
        # which we indicated with Z value of -1
        # Should ignore this when counting states
        uniqueZ = uniqueZ[uniqueZ >= 0]
    assert len(Mu) == uniqueZ.size
    return Z, Mu, np.asarray(Lscores)

def initKMeans_BregmanDiv_existing(
        X, K, obsModel,
        W=None,
        seed=0,
        smoothFrac=1.0,
        distexp=1.0):
    ''' Initialize cluster means Mu with existing clusters and K new clusters.

    Returns
    -------
    chosenZ : 1D array, size K
        int ids of atoms selected
    Mu : list of size Kexist + K
        each entry is a tuple of ND arrays
    minDiv : 1D array, size N
    '''
    PRNG = np.random.RandomState(int(seed))
    N = X.shape[0]
    if W is None:
        W = np.ones(N)

    # Create array to hold chosen data atom ids
    chosenZ = np.zeros(K, dtype=np.int32)

    # Initialize list Mu to hold all mean vectors
    # First obsModel.K entries go to existing clusters found in the obsModel.
    # Final K entries are placeholders for the new clusters we'll make below.
    Mu = [obsModel.getSmoothedMuForComp(k) for k in obsModel.K]
    Mu.extend([None for k in range K])

    # Compute minDiv between all data and existing clusters
    minDiv, DivDataVec = obsModel.calcSmoothedBregDiv(
        X=X, Mu=Mu[:obsModel.K], W=W,
        returnDivDataVec=True,
        return1D=True,
        smoothFrac=smoothFrac)

    # Sample each cluster id using distance heuristic
    for k in range(0, K):
        sum_minDiv = np.sum(minDiv)        
        if sum_minDiv == 0.0:
            # Duplicate rows corner case
            # Some rows of X may be exact copies, 
            # leading to all minDiv being zero if chosen covers all copies
            chosenZ = chosenZ[:k]
            for emptyk in reversed(range(k, K)):
                # Remove remaining entries in the Mu list,
                # so its total size is now k, not K
                Mu.pop(emptyk)
            # Escape loop to return statement below
            break

        elif sum_minDiv < 0 or not np.isfinite(sum_minDiv):
            raise ValueError("sum_minDiv not valid: %f" % (sum_minDiv))

        if distexp >= 9:
            chosenZ[k] = np.argmax(minDiv)
        else:
            if distexp > 1:
                minDiv = minDiv**distexp
                sum_minDiv = np.sum(minDiv)
            pvec = minDiv / sum_minDiv
            chosenZ[k] = PRNG.choice(N, p=pvec)

        # Compute mean vector for chosen data atom
        # Then add to the list
        Mu_k = obsModel.calcSmoothedMu(X[chosenZ[k]], W=W[chosenZ[k]])
        Mu[obsModel.K + k] = Mu_k

        # Performe distance calculation for latest chosen mean vector
        curDiv = obsModel.calcSmoothedBregDiv(
            X=X, Mu=Mu_k, W=W,
            DivDataVec=DivDataVec,
            return1D=True,
            smoothFrac=smoothFrac)
        # Enforce chosen data atom has distance 0
        # so we cannot pick it again
        curDiv[chosenZ[k]] = 0
        # Update distance between each atom and its nearest cluster
        minDiv = np.minimum(minDiv, curDiv)
    
    # Some final verification
    assert len(Mu) == chosenZ.size + obsModel.K
    return chosenZ, Mu, minDiv, np.sum(DivDataVec)
