'''
Monks.py

TODO : WHICH OF THE MATRICIES FROM THE FULL DATASET IS THIS??

The dataset was gathered during a perdio of political turmoil in the cloister.
The true labels (TrueZ) reflect the "faction labels" of each monk:
    0 = Young Turks (rebel group)
    1 = Loyal Opposition (monks who followed tradition and remained loyal),
    2 = Outcasts (Monks who were not accepted by either faction),
    3 = Waverers (Monks who couldn't decide on a group).

Resources
---------
Full version of dataset contains more relationships than the ones here.
http://vlado.fmf.uni-lj.si/pub/networks/data/ucinet/ucidata.htm#sampson
'''

import numpy as np
import scipy.io
import os

from bnpy.data import GraphXData


# Get path to the .mat file with the data
datasetdir = os.path.sep.join(
    os.path.abspath(__file__).split(
        os.path.sep)[
            :-
            1])
if not os.path.isdir(datasetdir):
    raise ValueError('CANNOT FIND MONKS DATASET DIRECTORY:\n' + datasetdir)

matfilepath = os.path.join(datasetdir, 'rawData', 'Monks.mat')
if not os.path.isfile(matfilepath):
    raise ValueError('CANNOT FIND MONKS DATASET MAT FILE:\n' + matfilepath)


def get_data(**kwargs):
    Data = GraphXData.read_from_mat(matfilepath)
    Data.summary = get_data_info()
    Data.name = get_short_name()
    return Data


def get_data_info():
    return 'Sampson Monks dataset'


def get_short_name():
    return 'Monks'


def summarize_data():
    Data = GraphXData.read_from_mat(matfilepath)
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

    print 'True community proportions = \n', Nsingle / np.sum(Nsingle)
    print 'True number of pairwise interactions with observed edge = \n', Npair
    print np.sum(Npair)

    import matplotlib.pyplot as plt
    from bnpy.viz import RelationalViz as relviz

    fig, ax = plt.subplots(1)
    relviz.plotNpair(Npair, ax, fig, title='True Npair Matrix')
    fig2, ax2 = plt.subplots(1)
    relviz.drawGraph(Data, ax2, fig2, Z, title='Graph With True Labels',
                     cmap='Set2')
    from IPython import embed
    embed()
    from sklearn.cluster import spectral_clustering
    A = Data.X.reshape((18, 18))
    labels = spectral_clustering(A, 4, 6)
    fig3, ax3 = plt.subplots(1)
    relviz.drawGraph(
        Data,
        ax3,
        fig3,
        colors=labels,
        title='Spectral Clustering')
    plt.show()

if __name__ == '__main__':
    summarize_data()
