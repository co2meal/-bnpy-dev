import scipy.io
import os
import matplotlib.pyplot as plt
import numpy as np
import imp
import sys
import argparse
import networkx as nx

import bnpy

from bnpy.util.StateSeqUtil import convertStateSeq_MAT2list
from bnpy.ioutil import BNPYArgParser
from bnpy.viz.TaskRanker import rankTasksForSingleJobOnDisk
from bnpy.viz import TaskRanker


def permuteTransMtx(A, Z):
    perms = np.array([])
    for k in xrange(np.max(Z) + 1):
        perms = np.append(perms, np.where(Z == k))

    A = A[perms, :]
    A = A[:, perms]
    return A


def plotNpair(Npair, curAx, fig, cmap='gray', title=''):
    plt.figure(fig.number)
    im = curAx.imshow(Npair / np.sum(Npair), cmap=cmap, vmin=0,
                      interpolation='nearest')
    Kmax = np.shape(Npair)[0] - 1
    curAx.set_xlim([-0.5, Kmax + 0.5])
    curAx.set_ylim([-0.5, Kmax + 0.5])
    curAx.set_yticks(np.arange(Kmax + 1))
    curAx.set_xticks(np.arange(Kmax + 1))
    curAx.invert_yaxis()
    vmax = np.max(Npair / np.sum(Npair))
    cbar = fig.colorbar(im, ticks=[0, vmax])
    curAx.set_title(title)


def plotEpi(jobnames, Data, laps=['final'], doBlockPerm=True, doShowTrue=True):
    print '*** NOTE *** plotEpi is ignoring your jobnames right now'
    Epi = Data.TrueParams['pi']
    estZ = np.argmax(Epi, axis=1)

    if doBlockPerm:  # Permute by most likely community
        perms = np.array([])
        for k in xrange(np.max(estZ) + 1):
            perms = np.append(perms, np.where(estZ == k))
        Epi = Epi[perms.astype(int)]

    fig, ax = plt.subplots(1)
    ax.imshow(Epi, cmap='Greys', interpolation='nearest')


def drawGraph(Data, curAx, fig, colors='r', cmap='gist_rainbow', title='',
              labels=None, edgeColors='k', edgeCmap=plt.cm.Greys):
    N = Data.nNodes
    if labels is None:
        labels = np.arange(N)

    if not hasattr(Data, 'edgeSet'):
        assert len(np.unique(Data.X)) <= 2 # binary
        inds = np.where(Data.X == 1)
        edgeSet=set(zip(inds[0], inds[1]))
        Data.edgeSet = edgeSet

    G = nx.DiGraph()
    G.add_nodes_from(np.arange(Data.nNodes))
    for e in Data.edgeSet:
      G.add_edge(e[0], e[1])

    plt.figure(fig.number)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=cmap)
    nx.draw_networkx_labels(G, pos, labels=dict(zip(np.arange(N), labels)))
    nx.draw_networkx_edges(G, pos, edge_color=edgeColors, edge_cmap=edgeCmap,
                           edge_vmin=min(edgeColors),
                           edge_vmax=max(edgeColors))

    # Beautification step
    # (turn off axes, trim window size, apply lipstick, etc)
    curAx.get_xaxis().set_visible(False)
    curAx.get_yaxis().set_visible(False)
    curAx.set_frame_on(False)
    cut = 1.05
    xmax = max(xx for xx, yy in pos.values())
    ymax = max(yy for xx, yy in pos.values())
    ygap = cut * ymax - ymax
    xgap = cut * xmax - xmax
    plt.xlim(0 - xgap, xmax + xgap)
    plt.ylim(0 - ygap, ymax + ygap)


def drawGraphVariationalDist(Data, Epi, curAx, fig, colors=None, labels=None,
                             cmap='gist_rainbow', title='', thresh=0.7,
                             seed=1234,
                             colorEdges=False, edgeCmap=plt.cm.Greys):
    np.random.seed(seed)
    G = nx.Graph()
    N = Data.nNodes
    G.add_nodes_from(np.arange(N))
    if colorEdges:
        edgeColors = list()
    else:
        edgeColors = 'k'

    for i in xrange(N):
        for j in xrange(N):
            if i == j or i > j:
                continue
            varDist = 1.0 - np.sum(np.abs(Epi[i] - Epi[j])) / 2.0
            if varDist > thresh:
                G.add_edge(i, j)
                if colorEdges:
                    edgeColors.append(varDist)

    if labels is not None:
        labels = dict(zip(np.arange(N), labels))

    plt.figure(fig.number)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=cmap)
    nx.draw_networkx_labels(G, pos, labels=labels)
    nx.draw_networkx_edges(G, pos, edge_color=edgeColors, edge_cmap=edgeCmap,
                           edge_vmin=min(edgeColors),
                           edge_vmax=max(edgeColors))
    curAx.get_xaxis().set_visible(False)
    curAx.get_yaxis().set_visible(False)
    curAx.set_frame_on(False)

    # Trim the frame to fit tightly around the graph
    cut = 1.05
    xmax = max(xx for xx, yy in pos.values())
    ymax = max(yy for xx, yy in pos.values())
    ygap = cut * ymax - ymax
    xgap = cut * xmax - xmax
    plt.xlim(0 - xgap, xmax + xgap)
    plt.ylim(0 - ygap, ymax + ygap)


def drawGraphEdgePr(Data, Epi, Ew, curAx, fig, colors=None, labels=None,
                    cmap='gist_rainbow', title='', thresh=0.7, seed=1234):
    np.random.seed(seed)
    G = nx.Graph()
    N = Data.nNodes
    G.add_nodes_from(np.arange(N))
    for i in xrange(N):
        for j in xrange(N):
            if i == j:
                continue
            pr = np.sum(
                Epi[i, np.newaxis, :, np.newaxis] *
                Epi[np.newaxis, j, np.newaxis, :] * Ew)
            if pr > thresh:
                G.add_edge(i, j)

    plt.figure(fig.number)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=cmap)

    if labels is None:
        labels = np.arange(N)
    else:
        labels = ['%.2f' % l for l in labels]
    nx.draw_networkx_labels(G, pos, labels=dict(zip(np.arange(N), labels)))
    nx.draw_networkx_edges(G, pos)

    curAx.get_xaxis().set_visible(False)
    curAx.get_yaxis().set_visible(False)
    curAx.set_frame_on(False)

    # Trim the frame to fit tightly around the graph
    cut = 1.05
    xmax = max(xx for xx, yy in pos.values())
    ymax = max(yy for xx, yy in pos.values())
    ygap = cut * ymax - ymax
    xgap = cut * xmax - xmax
    plt.xlim(0 - xgap, xmax + xgap)
    plt.ylim(0 - ygap, ymax + ygap)


def plotSingleJob(dataset, jobname, taskids='1', lap='final',
                  showELBOInTitle=True, cmap='gray', title='', mixZs=False):
    Datamod = imp.load_source(
        dataset,
        os.path.expandvars('$BNPYDATADIR/' + dataset + '.py'))
    Data = Datamod.get_data()
    jobpath = os.path.join(os.path.expandvars('$BNPYOUTDIR'),
                           Datamod.get_short_name(), jobname)

    if isinstance(taskids, str):
        taskids = BNPYArgParser.parse_task_ids(jobpath, taskids)
    elif isinstance(taskids, int):
        taskids = [str(taskids)]

    #f_pair, axes_pair = plt.subplots(1, len(taskids))
    #f_graph, axes_graph = plt.subplots(1, len(taskids))
    #f_vardist, axes_vardist = plt.subplots(1, len(taskids), frameon=False)
    #if len(taskids) == 1:
    #    axes_pair = [axes_pair]

    AdjMat = np.squeeze(Data.toAdjacencyMatrix())
    sortids = np.argsort(Data.TrueParams['pi'].argmax(axis=1))
    AdjMat = AdjMat[sortids, :]
    AdjMat = AdjMat[:, sortids]


    # Show the true adj mat and the estimated side-by-side
    ncols = len(taskids)+1
    plt.subplots(nrows=1, ncols=ncols, figsize=(3, 3*ncols))
    plt.subplot(1, ncols, 1)
    plt.imshow(AdjMat, cmap='Greys', interpolation='nearest', vmin=0, vmax=1)

    for tt, taskid in enumerate(taskids):
        # curAx = axes_pair[tt]
        taskoutpath = os.path.join(jobpath, taskid) + os.path.sep

        # Figure out which lap to use
        if lap == 'final':
            lapsFile = open(taskoutpath + 'laps.txt')
            curLap = lapsFile.readlines()
            curLap = float(curLap[-1])
            lapsFile.close()
        else:
            curLap = int(lap)

        hmodel, curLap = bnpy.ioutil.ModelReader.loadModelForLap(
            taskoutpath, curLap)
        Ew = hmodel.obsModel.Post.lam1 / \
            (hmodel.obsModel.Post.lam1 + hmodel.obsModel.Post.lam0)
        LP = hmodel.calc_local_params(Data)
        taskAdjMat = np.zeros((Data.nNodes, Data.nNodes, Data.dim))
        for s in range(Data.nNodes):
            for t in range(Data.nNodes):
                if s == t:
                    continue
                eid = np.flatnonzero(
                    np.logical_and(Data.edges[:,0] == s, 
                                   Data.edges[:,1] == t))[0]
                resp_st = LP['resp'][eid, :, :]
                taskAdjMat[s,t] = np.sum(resp_st[:,:,np.newaxis] * Ew, axis=(0,1))
                assert np.allclose(resp_st.sum(), 1.0)

        assert taskAdjMat.min() >= 0
        assert taskAdjMat.max() <= 1.0
        taskAdjMat = np.squeeze(taskAdjMat)
        taskAdjMat = taskAdjMat[sortids,:]
        taskAdjMat = taskAdjMat[:, sortids]

        plt.subplot(1, ncols, 2+tt)
        plt.imshow(taskAdjMat,
                   cmap='Greys', interpolation='nearest', vmin=0, vmax=1)
        
        print 'ERROR: ', np.sum(AdjMat - taskAdjMat)
        '''
        if showELBOInTitle:
            ELBOscores = np.loadtxt(os.path.join(path, 'evidence.txt'))
            laps = np.loadtxt(os.path.join(path, 'laps.txt'))
            savedLaps = np.loadtxt(os.path.join(path, 'laps-saved-params.txt'))

            loc = np.argmin(np.abs(laps - curLap))
            ELBO = ELBOscores[loc]
            title = title + ' ELBO: %.3f' % ELBO

        # Plot the Npair matrix
        filename = 'Lap%08.3fSuffStats.mat' % (curLap)
        Npair = scipy.io.loadmat(path + filename)
        Npair = Npair['Npair']
        plotNpair(Npair, curAx, f_pair, cmap=cmap, title=title)

        # Draw network graph, colored by estimated communities
        filename = 'Lap%08.3fAllocModel.mat' % (curLap)
        amod = scipy.io.loadmat(path + filename)
        estZ = amod['estZ']
        if mixZs:  # used to
            theta = amod['theta']
            pi = theta / np.sum(theta, axis=1)[:, np.newaxis]
            K = pi.shape[1]
            N = pi.shape[0]
            zs = np.asarray([float(z) for z in np.arange(K)])
            estZ = np.tile(zs, (N, 1))
            estZ *= pi
            estZ = np.sum(estZ, axis=1)
            estZ /= np.max(estZ)
        drawGraph(Data, axes_graph, f_graph, estZ, title=title)
        axes_graph.set_title(title)

        # Plot graph with edges based on variational dist. of pi_i and pi_j
        Epi = amod['theta']
        Epi /= np.sum(Epi, axis=1)[:, np.newaxis]
        drawGraphVariationalDist(Data, Epi, axes_vardist, f_vardist, Z=estZ,
                                 title=title)
        '''


def plotTrueLabels(dataset, Data=None, gtypes=['Actual'], thresh=.5,
                   mixColors=False, colorEdges=False, title=''):
    if Data is None:
        Datamod = imp.load_source(
            dataset,
            os.path.expandvars('$BNPYDATADIR/' + dataset + '.py'))
        Data = Datamod.get_data()

    if mixColors:
        Epi = Data.TrueParams['pi']
        K = np.shape(Epi)[1]
        colors = np.sum(Epi * np.arange(K)[np.newaxis, :], axis=1)
    else:
        if Data.TrueParams['Z'].ndim == 1:  # single membership model
            colors = Data.TrueParams['Z']
        else:
            colors = np.argmax(Data.TrueParams['pi'], axis=1)

    if Data.TrueParams['pi'].ndim == 2:
        pi = Data.TrueParams['pi']

    for gtype in gtypes:
        fig, ax = plt.subplots(1)
        if gtype == 'Actual':
            if title is None:
                title = dataset + 'True Labels'
            drawGraph(Data, curAx=ax, fig=fig, colors=colors,
                      title=title)

        elif gtype == 'VarDist':
            if title is None:
                title = dataset + 'True Labels'
            drawGraphVariationalDist(Data, pi, ax, fig, colors=colors,
                                     title=title,
                                     thresh=thresh, colorEdges=colorEdges)

        elif gtype == 'EdgePr':
            drawGraphEdgePr(Data, pi, Data.TrueParams['w'], ax, fig,
                            colors=colors, title=title, thresh=thresh)


def plotTransMtxTruth(Data, perms=None, gtypes=['Actual'], doPerm=False,
                      thresh=.22):

    pi = Data.TrueParams['pi']
    Z = np.argmax(pi, axis=1)
    N = Data.nNodes
    K = np.max(Z) + 1

    if doPerm:
        if perms is None:
            perms = np.array([])
            for k in xrange(np.max(Z) + 1):
                perms = np.append(perms, np.where(Z == k))
            perms = perms.astype(int)

    for gtype in gtypes:
        if gtype == 'Actual':
            plotActualTransMtx(Data, perms, doPerm)
        elif gtype == 'EdgePr':
            phi = loadTrueObsParams(Data)
            plotEdgePrTransMtx(Data, pi, phi, perms, doPerm)


def loadTrueObsParams(Data):
    if 'w' in Data.TrueParams:
        phi = np.asarray(Data.TrueParams['w'])
    elif 'sigma' in Data.TrueParams and 'mu' in Data.TrueParams:
        phi = np.asarray(Data.TrueParams['mu'])
        if phi.ndim == 1:  # Assortative case
            K = len(phi)
            mu = np.zeros((K, K))
            mu[np.diag_indices(K)] = phi
            phi = mu

    else:
        raise NotImplementedError(
            'DATA TYPE NOT SUPPORTED BY RELATIONALVIZ.PY')
    return phi


def plotTransMtxEst(Data, dataset, jobnames, perms=None,
                    gtypes=['Actual'], doPerm=False, thresh=.22):

    for jobname in jobnames:
        jobpath = os.path.join(os.path.expandvars('$BNPYOUTDIR'),
                               dataset, jobname)
        ranks = TaskRanker.rankTasksForSingleJob(jobpath)
        print jobpath
        amod = scipy.io.loadmat(os.path.join(jobpath, str(ranks[0]),
                                             'BestAllocModel.mat'))
        aprior = scipy.io.loadmat(os.path.join(jobpath, str(ranks[0]),
                                               'AllocPrior.mat'))
        omod = scipy.io.loadmat(os.path.join(jobpath, str(ranks[0]),
                                             'BestObsModel.mat'))
        ss = scipy.io.loadmat(os.path.join(jobpath, str(ranks[0]),
                                           'BestSuffStats.mat'))

        # Get expectations under q() of model parameters
        Epi = amod['theta']
        Epi /= np.sum(Epi, axis=1)[:, np.newaxis]
        Ephi = computeMeanObsParams(omod, aprior)
        if Ephi.shape[0] != Epi.shape[1]:
            Epi = Epi[:, :-1]

        Z = np.argmax(Epi, axis=1)
        N = Epi.shape[0]
        K = np.max(Z) + 1

        if doPerm:
            if perms is None:
                curPerms = np.array([])
                for k in xrange(np.max(Z) + 1):
                    curPerms = np.append(curPerms, np.where(Z == k))
                curPerms = curPerms.astype(int)
            else:
                curPerms = perms
        else:
            curPerms = None

        for gtype in gtypes:
            if gtype == 'Actual':
                plotActualTransMtx(Data, curPerms, doPerm)
            elif gtype == 'EdgePr':
                print '--------'
                print dataset + '/' + jobname
                print ranks[0]
                prs = plotEdgePrTransMtx(Data, Epi, Ephi, curPerms, doPerm,
                                         title=jobname,
                                         true=None)
                scipy.io.savemat(dataset + '-' + jobname + '.mat',
                                 {'prs': prs, 'perms': curPerms})

        # plt.show()


def computeMeanObsParams(omod, aprior):

    if omod['name'][0].count('Bern') > 0:
        tau1 = omod['lam1']
        tau0 = omod['lam0']
        Ephi = np.squeeze(tau1 / (tau1 + tau0))

        if Ephi.ndim == 1:  # Assortative case
            K = len(Ephi)
            phi = np.ones((K, K)) * aprior['epsilon']
            phi[np.diag_indices(K)] = Ephi
            Ephi = phi

    elif omod['name'][0].count('Gauss') > 0:
        Ephi = np.squeeze(omod['m'])

        if Ephi.ndim == 1:  # Assortative case
            K = len(Ephi)
            mu = np.zeros((K, K))
            mu[np.diag_indices(K)] = Ephi
            Ephi = mu

    return Ephi


def plotActualTransMtx(Data, perms=None, doPerm=True):

  if Data.isSparse:
    G = nx.DiGraph()
    G.add_nodes_from(np.arange(Data.nNodes))
    for e in Data.edgeSet:
      G.add_edge(e[0], e[1])
    A = nx.to_numpy_matrix(G)
  else:
    A = np.squeeze(Data.Xmatrix[:,:,0])

  if doPerm:
    print 'do perm'
    if perms is not None:
      print 'ok im here'
      A = A[perms, :]
      A = A[:, perms]

  fig, ax = plt.subplots(1)
  ax.imshow(A, cmap='Greys', interpolation='nearest')


def plotEdgePrTransMtx(Data, pi, phi, perms, doPerm, title='', true=None):
    N = pi.shape[0]
    prs = np.zeros((N, N))

    for i in xrange(N):
        for j in xrange(N):
            if i == j:
                continue
            prs[i,
                j] = np.sum(pi[i,
                               np.newaxis,
                               :,
                               np.newaxis] * pi[np.newaxis,
                                                j,
                                                np.newaxis,
                                                :] * phi)

    if doPerm:
        prs = prs[perms, :]
        prs = prs[:, perms]

    fig, ax = plt.subplots(1)
    if true is not None:
        ax.imshow(prs - true, cmap='coolwarm_r', interpolation='nearest',
                  vmin=-1.0, vmax=1.0)
        pass
    else:
        ax.imshow(prs, cmap='Greys', interpolation='nearest')
        pass

    # ax.set_title(title)
    return prs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataName')
    parser.add_argument('jobname')
    parser.add_argument(
        '--taskids', type=str, default='1',
        help="int ids of tasks (trials/runs) to plot from given job." +
        " Example: '4' or '1,2,3' or '2-6'.")

    parser.add_argument('--lap', default='final')
    args = parser.parse_args()

    plotSingleJob(dataset=args.dataName,
                  jobname=args.jobname,
                  taskids=args.taskids,
                  lap=args.lap)
    plt.show()


def getEstZ(jobnames, dataset):
    zdict = dict()
    pidict = dict()
    for jobname in jobnames:
        jobpath = os.path.join(os.path.expandvars('$BNPYOUTDIR'),
                               dataset, jobname)
        ranks = TaskRanker.rankTasksForSingleJob(jobpath)

        amod = scipy.io.loadmat(os.path.join(jobpath, str(ranks[0]),
                                             'BestAllocModel.mat'))
        aprior = scipy.io.loadmat(os.path.join(jobpath, str(ranks[0]),
                                               'AllocPrior.mat'))
        omod = scipy.io.loadmat(os.path.join(jobpath, str(ranks[0]),
                                             'BestObsModel.mat'))
        ss = scipy.io.loadmat(os.path.join(jobpath, str(ranks[0]),
                                           'BestSuffStats.mat'))

        # Get expectations under q() of model parameters
        Epi = amod['theta']
        Epi /= np.sum(Epi, axis=1)[:, np.newaxis]
        tau1 = omod['lam1']
        tau0 = omod['lam0']
        Ew = np.squeeze(tau1 / (tau1 + tau0))

        if Ew.shape[0] != Epi.shape[1]:
            Epi = Epi[:, :-1]

        if Ew.ndim == 1:  # Assortative case
            K = len(Ew)
            w = np.ones((K, K)) * aprior['epsilon']
            w[np.diag_indices(K)] = Ew
            Ew = w

        zdict[jobname] = np.argmax(Epi, axis=1)
        pidict[jobname] = Epi

    return zdict, pidict
