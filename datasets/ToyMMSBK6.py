'''
ToyMMSBK6.py
'''

import numpy as np
from bnpy.data import GraphXData

K = 6


def get_data(
        seed=123, nNodes=100, alpha=0.05,
        w_diag=.8,
        w_offdiag_eps=.01,
        **kwargs):
    ''' Create toy dataset as bnpy GraphXData object.

    Args
    -------
    seed : int
        seed for random number generator
    nNodes : int
        number of nodes in the generated network

    Returns
    -------
    Data : bnpy GraphXData object
    '''
    prng = np.random.RandomState(seed)

    # Create membership probabilities at each node
    if not hasattr(alpha, '__len__'):
        alpha = alpha * np.ones(K)
    pi = prng.dirichlet(alpha, size=nNodes)

    # Create block relation matrix W, shape K x K
    w = w_offdiag_eps * np.ones((K, K))
    w[np.diag_indices(6)] = w_diag

    # Generate community assignments, s, r, and pack into TrueZ
    s = np.zeros((nNodes, nNodes), dtype=int)
    r = np.zeros((nNodes, nNodes), dtype=int)
    for i in xrange(nNodes):
        s[i, :] = prng.choice(xrange(K), p=pi[i, :], size=nNodes)
        r[:, i] = prng.choice(xrange(K), p=pi[i, :], size=nNodes)
    TrueZ = np.zeros((nNodes, nNodes, 2), dtype=int)
    TrueZ[:, :, 0] = s
    TrueZ[:, :, 1] = r

    TrueParams = dict(Z=TrueZ, w=w, pi=pi)

    # Generate edge set
    sourceID = list()
    destID = list()
    for i in xrange(nNodes):
        for j in xrange(nNodes):
            if i == j:
                continue
            y_ij = prng.binomial(n=1, p=w[s[i, j], r[i, j]])
            if y_ij == 1:
                sourceID.append(i)
                destID.append(j)
    EdgeSet = set(zip(sourceID, destID))

    Data = GraphXData(X=None, edgeSet=EdgeSet,
                      nNodesTotal=nNodes, nNodes=nNodes,
                      TrueParams=TrueParams, isSparse=True)
    Data.name = get_short_name()
    return Data


def get_short_name():
    return 'ToyMMSBK6'


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from bnpy.viz import RelationalViz

    Data = get_data(nNodes=100)
    w = Data.TrueParams['w']

    # RelationalViz.plotTrueLabels(
    #    'ToyMMSBK6', Data,
    #    gtypes=['Actual', 'VarDist', 'EdgePr'],
    #    mixColors=True, thresh=.73, colorEdges=False, title='')

    # Plot subset of pi
    Epi = Data.TrueParams['pi']
    fix, ax = plt.subplots(1)
    ax.imshow(
        Epi[0:30, :],
        cmap='Greys', interpolation='nearest',
        vmin=0, vmax=1.0)
    ax.set_ylabel('nodes')
    ax.set_xlabel('states')
    ax.set_title('Membership vectors pi')

    # Plot w
    fig, ax = plt.subplots(1)
    im = ax.imshow(
        w, cmap='Greys', interpolation='nearest',
        vmin=0, vmax=1.0)
    ax.set_xlabel('states')
    ax.set_xlabel('states')
    ax.set_title('Edge probability matrix w')

    plt.show()
