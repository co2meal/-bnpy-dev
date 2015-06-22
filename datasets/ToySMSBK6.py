import numpy as np
from bnpy.data import GraphXData


def get_data(seed=123, nNodes=80, K=6, alpha=1.0,
             tau1=1.0, tau0=10.0, genSparse=True, **kwargs):
    ''' Generate several data sequences, returned as a bnpy data-object

    Args
    -------
    seed : integer seed for random number generator,

    Returns
    -------
    Data : bnpy GraphXData object, with nObsTotal observations
    '''
    global g_K, g_tau1, g_tau0, g_alpha
    g_K = K
    g_tau1 = tau1
    g_tau0 = tau0
    g_alpha = alpha

    X, sourceID, destID, Z, w, pi = \
        gen_data(seed, nNodes, K, alpha, tau1, tau0, genSparse=genSparse)
    TrueParams = dict(Z=Z, w=w, pi=pi)
    adjList = np.tile(np.arange(nNodes), (nNodes, 1))
    if genSparse:
        X = None
    else:
        X = np.ravel(X).T
    data = GraphXData(X=X, sourceID=sourceID,
                      destID=destID, nNodesTotal=nNodes, nNodes=nNodes,
                      TrueParams=TrueParams, isSparse=genSparse)
    return data


def get_short_name():
    global g_K, g_tau1, g_tau0, g_alpha
    return 'ToySMSBK6'


def get_data_info():
    global g_K, g_tau1, g_tau0, g_alpha
    return 'Toy SMSB dataset. K=%d' % (g_K)
    # return 'Toy data generated from a single membership stochastic block
    # model with K=%d communities, pi~Dir(%.2f), w_{kl} ~ Beta(%.2f, %.2f)' % (
    # K, alpha, tau1, tau0)


w = np.asarray([
    [.6, .10, .01, .05, .01, .01],
    [.01, .6, .10, .01, .01, .01],
    [.01, .01, .6, .10, .01, .01],
    [.01, .01, .01, .6, .10, .01],
    [.01, .01, .01, .01, .6, .10],
    [.10, .01, .01, .01, .01, .6]
])


def gen_data(seed=123, nNodes=80, K=6, alpha=20.0,
             tau1=3.0, tau0=1.0, genSparse=True, **kwargs):
    if not hasattr(alpha, '__len__'):
        alpha = alpha * np.ones(K)

    prng = np.random.RandomState(seed)
    pi = prng.dirichlet(alpha)

    if not genSparse:
        z = prng.choice(xrange(K), p=pi, size=nNodes)
        x = np.zeros((nNodes, nNodes))
        for i in xrange(nNodes):
            for j in xrange(nNodes):
                if i == j:
                    x[i, j] = 0
                else:
                    x[i, j] = prng.binomial(n=1, p=w[z[i], z[j]])

        adjList = np.tile(np.arange(nNodes), (nNodes, 1))
        sourceID = np.ravel(adjList.T)
        destID = np.ravel(adjList)

    else:
        z = prng.choice(xrange(K), p=pi, size=nNodes)
        sourceID = list()
        destID = list()
        for i in xrange(nNodes):
            for j in xrange(nNodes):
                if i == j:
                    continue
                else:
                    if prng.binomial(n=1, p=w[z[i], z[j]]) == 1:
                        sourceID.append(i)
                        destID.append(j)

        x = None
    print pi
    return x, sourceID, destID, z, w, pi


def summarize_data(Data):
    Z = Data.TrueParams['Z']
    N = Data.nNodes
    zMax = len(np.unique(Z))
    Npair = np.zeros((zMax, zMax))
    Nsingle = np.zeros(zMax)

    for i in xrange(N):
        Nsingle[Z[i]] += 1
        for j in xrange(N):
            if i == j:
                continue
            for l in xrange(zMax):
                for m in xrange(zMax):
                    if Z[i] == l and Z[j] == m:
                        if Data.X[i * N + j] == 1:
                            Npair[l, m] += 1

    X = Data.X.reshape((N, N))
    deg = np.sum(X, axis=1) + np.sum(X, axis=0)
    cnts = np.bincount(deg.astype(int))
    fig, ax = plt.subplots(1)
    ax.scatter(np.arange(len(cnts)), cnts)
    print 'True community proportions = \n', Nsingle / np.sum(Nsingle)
    print 'True number of pairwise interactions with observed edge = \n', Npair
    print np.sum(Npair)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from bnpy.viz import RelationalViz as relviz
    data = get_data(nNodes=500, genSparse=True)
    # summarize_data(data)
    # print np.sum(data.X)
    f, ax = plt.subplots(1)
    relviz.drawGraph(data, fig=f, curAx=ax, colors=data.TrueParams['Z'],)

    f, ax = plt.subplots(1)
    ax.imshow(data.X.reshape(data.nNodes, data.nNodes))
    plt.show()
