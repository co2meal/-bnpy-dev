'''
PrintTopics.py

Prints the top topics

Usage
-------
python PrintTopics.py dataName allocModelName obsModelName algName [options]

Saves topics as top_words.txt within the directory that the script draws from.

Options
--------
--topW : int
    Desired number of top words to show.
    (Must be less than the size of vocabulary).
--taskids : int or array_like
    ids of the tasks (individual runs) of the given job to plot.
    Ex: "1" or "3" or "1,2,3" or "1-6"
'''
from matplotlib import pylab
import numpy as np
import argparse
import os
import sys

import bnpy
from bnpy.ioutil.ModelReader import loadWordCountMatrixForLap, loadModelForLap

STYLE = """
<style>
pre {line-height:13px; font-size:13px; display:inline; color:black;}
h2 {line-height:16px; font-size:16px; color:gray;
    text-align:center; padding:0px; margin:0px;}
td {padding-top:5px; padding-bottom:5px;}
</style>
"""


def showTopWordsForTask(taskpath, vocabfile, lap=None, doHTML=1,
                        doCounts=1, **kwargs):
    ''' Print top words for each topic from results saved on disk.

    Returns
    -------
    html : string, ready-to-print to display of the top words
    '''
    with open(vocabfile, 'r') as f:
        vocabList = [x.strip() for x in f.readlines()]

    if doCounts and (lap is None or lap > 0):
        WordCounts = loadWordCountMatrixForLap(taskpath, lap)
        countVec = WordCounts.sum(axis=1)
        if doHTML:
            return htmlTopWordsFromWordCounts(
                WordCounts, vocabList, countVec=countVec, **kwargs)
        else:
            return printTopWordsFromWordCounts(WordCounts, vocabList)

    else:
        hmodel, lap = loadModelForLap(taskpath, lap)
        if doHTML:
            return htmlTopWordsFromHModel(hmodel, vocabList, **kwargs)
        else:
            return printTopWordsFromHModel(hmodel, vocabList)


def htmlTopWordsFromWordCounts(WordCounts, vocabList, order=None, Ktop=10,
                               ncols=5, maxKToDisplay=50, countVec=None,
                               activeCompIDs=None, **kwargs):
    K, W = WordCounts.shape
    if order is None:
        order = np.arange(K)
    if activeCompIDs is None:
        activeCompIDs = np.arange(K)
    if countVec is None:
        countVec = np.sum(WordCounts, axis=1)

    htmllines = list()
    htmllines.append(STYLE)
    htmllines.append('<table>')
    for posID, compID in enumerate(order[:maxKToDisplay]):
        if posID % ncols == 0:
            htmllines.append('  <tr>')

        k = np.flatnonzero(activeCompIDs == compID)
        if len(k) == 1:
            k = k[0]
            titleline = '<h2> %6d </h2>' % (countVec[k])
            htmllines.append('    <td>' + titleline)
            htmllines.append('    <pre>')
            topIDs = np.argsort(-1 * WordCounts[k])[:Ktop]
            for topID in topIDs:
                dataline = '%5d %s ' % (
                    WordCounts[k, topID], vocabList[topID][:16])
                htmllines.append(dataline)
            htmllines.append('    </pre></td>')
        else:
            htmllines.append('    <td></td>')

        if posID % ncols == ncols - 1:
            htmllines.append(' </tr>')
    htmllines.append('</table>')
    return '\n'.join(htmllines)


def htmlTopWordsFromHModel(hmodel, vocabList, order=None, Ktop=10,
                           ncols=5, maxKToDisplay=50, activeCompIDs=None,
                           **kwargs):
    try:
        topics = hmodel.obsModel.EstParams.phi
    except AttributeError:
        hmodel.obsModel.setEstParamsFromPost(hmodel.obsModel.Post)
        topics = hmodel.obsModel.EstParams.phi
    K, W = topics.shape
    if order is None:
        order = np.arange(K)
    if activeCompIDs is None:
        activeCompIDs = np.arange(K)

    htmllines = list()
    htmllines.append(STYLE)
    htmllines.append('<table>')

    for posID, compID in enumerate(order[:maxKToDisplay]):
        if posID % ncols == 0:
            htmllines.append('  <tr>')

        k = np.flatnonzero(activeCompIDs == compID)
        if len(k) == 1:
            k = k[0]
            htmllines.append('    <td><pre>')
            topIDs = np.argsort(-1 * topics[k])[:Ktop]
            for topID in topIDs:
                dataline = ' %.3f %s ' % (
                    topics[k, topID], vocabList[topID][:16])
                htmllines.append(dataline)
            htmllines.append('    </pre></td>')

        else:
            htmllines.append('   <td></td>')

        if posID % ncols == ncols - 1:
            htmllines.append(' </tr>')
    htmllines.append('</table>')
    return '\n'.join(htmllines)


def printTopWordsFromHModel(hmodel, vocabList, **kwargs):
    try:
        topics = hmodel.obsModel.EstParams.phi
    except AttributeError:
        hmodel.obsModel.setEstParamsFromPost(hmodel.obsModel.Post)
        topics = hmodel.obsModel.EstParams.phi
    printTopWordsFromTopics(topics, vocabList, **kwargs)


def printTopWordsFromWordCounts(WordCounts, vocabList, order=None, Ktop=10):
    K, W = WordCounts.shape
    if order is None:
        order = np.arange(K)
    N = np.sum(WordCounts, axis=1)
    for posID, k in enumerate(order):
        print '----- topic %d. count %5d.' % (k, N[k])

        topIDs = np.argsort(-1 * WordCounts[k])
        for wID in topIDs[:Ktop]:
            print '%3d %s' % (WordCounts[k, wID], vocabList[wID])


def printTopWordsFromTopics(topics, vocabList, ktarget=None, Ktop=10):
    K, W = topics.shape
    if ktarget is not None:
        topIDs = np.argsort(-1 * topics[ktarget])
        for wID in topIDs[:Ktop]:
            print '%.3f %s' % (topics[ktarget, wID], vocabList[wID])
        return
    # Base case: print all topics
    for k in xrange(K):
        topIDs = np.argsort(-1 * topics[k])
        for wID in topIDs[:Ktop]:
            print '%.3f %s' % (topics[k, wID], vocabList[wID])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('taskpath')
    parser.add_argument('vocabfilepath')
    args = parser.parse_args()
    print showTopWordsForTask(args.taskpath, args.vocabfilepath)
