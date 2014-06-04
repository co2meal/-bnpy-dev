''' LocalStep routines, for StickBreak (SB) version of the HDP
'''

import numpy as np
from scipy.special import digamma, gammaln
import scipy.optimize
import os

from bnpy.util import NumericUtil, LibLocalStep
from bnpy.allocmodel.admix import OptimizerForMAPDocTopicSticks as MAPOptimizer

########################################################### doc-level stickbrk
###########################################################  version
def calcLocalDocParams(Data, LP, topicPrior1, topicPrior0, 
                             nCoordAscentItersLP=2,
                             nCoordAscentFromScratchLP=25,
                             convThrLP=0.0001,
                             methodLP='scratch',
                             doInPlaceLP=1,
                             logdirLP=None,
                             **kwargs):
  ''' Calculate local parameters for all documents, given topic prior

      Args
      -------

      Returns 
      -------
      LP : dictionary with fields
  '''
  D = Data.nDoc
  K = topicPrior1.size

  if 'expEloglik' in LP:
    expEloglik = LP['expEloglik']
  else:
    if doInPlaceLP:
      expEloglik = LP['E_logsoftev_WordsData']
    else:
      # Need to preserve E_logsoftev for the perDocELBO calc
      expEloglik = LP['E_logsoftev_WordsData'].copy()

    expEloglik -= expEloglik.max(axis=1)[:,np.newaxis] 
    NumericUtil.inplaceExp(expEloglik)
    LP['expEloglik'] = expEloglik
    if doInPlaceLP:
      del LP['E_logsoftev_WordsData'] # just remove the key

  ######## Allocate document-specific variables
  docptr = np.asarray(np.hstack([0, Data.doc_range[:,1]]), dtype=np.int32)
  hasCountOfCorrectSize = 'DocTopicCount' in LP \
                           and LP['DocTopicCount'].shape[1] == K 

  if methodLP.count('bounce'):
    nCoordAscentItersLP = nCoordAscentFromScratchLP

  if hasCountOfCorrectSize and methodLP.count('memo'):
    LP['DocTopicCount'] = LP['DocTopicCount'].copy() # local copy

    # Update U1, U0
    LP = update_U1U0_SB(LP, topicPrior1, topicPrior0, **kwargs)
    # Update expected value of log Pi[d,k]
    LP = update_ElogPi_SB(LP, **kwargs)
    if not methodLP == 'c':
      expElogpi = np.exp(LP['E_logPi'])
    else:
      expElogpi = np.empty((D,K), order='F')
      np.exp(LP['E_logPi'], out=expElogpi)
 
  else:
    nCoordAscentItersLP = nCoordAscentFromScratchLP

    if methodLP == 'c':
      LP['DocTopicCount'] = np.zeros((D, K), order='F')
    else:
      LP['DocTopicCount'] = np.zeros((D, K))

    # expElogpi is uniform over topics by default
    #  certain methods may modify this below
    expElogpi = np.ones_like(LP['DocTopicCount'])


  ######## Allocate token-specific variables
  # sumRTilde : nDistinctWords vector. row n = \sum_{k} \tilde{r}_{nk} 
  sumRTilde = np.zeros(Data.nObs)

  ######## Repeat updates until old_theta has stopped changing ...
  activeDocs = np.arange(D, dtype=np.int32)
  old_DocTopicCount = LP['DocTopicCount'].copy()

  for ii in xrange(nCoordAscentItersLP):

    # Update expElogpi for active documents
    if ii > 0:
      expElogpi[activeDocs] = np.exp(LP['E_logPi'][activeDocs])

    sumRTilde, LP['DocTopicCount'] = LibLocalStep.calcDocTopicCount(
                                       activeDocs, docptr,
                                       Data.word_count, expElogpi, expEloglik,
                                       sumRTilde, LP['DocTopicCount'],
                                       methodLP=methodLP,
                                     )

    # Update U1, U0
    LP = update_U1U0_SB(LP, topicPrior1, topicPrior0, **kwargs)

    # Update expected value of log Pi[d,k]
    LP = update_ElogPi_SB(LP, activeDocs, **kwargs)
    
    # Assess convergence
    docDiffs = np.max(np.abs(old_DocTopicCount - LP['DocTopicCount']), axis=1)
    if np.max(docDiffs) < convThrLP:
      break
    activeDocs = np.asarray(np.flatnonzero(docDiffs > convThrLP),
                            dtype=np.int32)

    # Store DocTopicCount for next round's convergence test
    # Here, the "[:]" syntax ensures we do NOT copy by reference
    old_DocTopicCount[activeDocs] = LP['DocTopicCount'][activeDocs]
    ### end loop over alternating-ascent updates

    if logdirLP and ii % 10 == 0:
      write_to_log(docDiffs, ii, 0, nCoordAscentItersLP, Data, methodLP, logdirLP)

  LP['didConverge'] = np.max(docDiffs) < convThrLP
  LP['maxDocDiff'] = np.max(docDiffs)
  LP['nCoordAscentIters'] = ii
  LP['sumRTilde'] = sumRTilde
  LP['expElogpi'] = expElogpi

  if logdirLP:
    write_to_log(docDiffs, ii, 1, nCoordAscentItersLP, Data, methodLP, logdirLP)
  return LP

def write_to_log(docDiffs, ii, isFinal, nCoordAscentItersLP, Data, methodLP, logdirLP):
    dataid = np.sum(Data.word_id[:3]) * np.sum( Data.doc_range[:3, 1])
    dataid = hash(dataid) % 1000
    if isFinal:
      filestr = 'LP-id%03d-%s.txt' % (dataid, methodLP)
    else:
      filestr = 'LP-id%03d-%s-%02d.txt' % (dataid, methodLP, ii+1)
    with open(os.path.join(logdirLP, filestr),'a') as f:
      line = '%3d %3d %8.3f %8.3f %8.3f %8.3f %5d\n' % (
                        ii+1,
                        nCoordAscentItersLP,
                        np.median(docDiffs),
                        np.percentile(docDiffs, 90),
                        np.percentile(docDiffs, 95),
                        np.max(docDiffs),
                        Data.nDoc
                        )

      f.write(line)


def write_method_wins_to_log(Data, methodLP, nWins, nTotal, logdirLP):
    dataid = np.sum(Data.word_id[:3]) * np.sum( Data.doc_range[:3, 1])
    dataid = hash(dataid) % 1000

    filestr = 'LP-id%03d-%s-wins.txt' % (dataid, methodLP)
    with open(os.path.join(logdirLP, filestr),'a') as f:
      line = '%5d %5d\n' % (nWins, nTotal)
      f.write(line)

########################################################### doc-level beta
########################################################### helpers
def update_U1U0_SB(LP, topicPrior1, topicPrior0,**kwargs):
  ''' Update document-level stick-breaking beta parameters, U1 and U0.
  '''
  assert 'DocTopicCount' in LP
  K =  LP['DocTopicCount'].shape[1]
  if 'U1' in LP and LP['U1'].shape == LP['DocTopicCount'].shape:
    # no new memory allocated here
    LP['U1'][:] = LP['DocTopicCount'] + topicPrior1
    calcDocTopicRemCount(LP['DocTopicCount'], out=LP['U0'][:,::-1])
    LP['U0'] += topicPrior0
  else:
    LP['U1'] = LP['DocTopicCount'] + topicPrior1
    LP['U0'] = calcDocTopicRemCount(LP['DocTopicCount']) + topicPrior0
  return LP

def update_ElogPi_SB(LP, activeDocs=None, **kwargs):
  ''' Update expected log topic appearance probabilities in each doc
  '''
  shp = LP['U1'].shape
  if 'digammaBoth' not in LP or shp != LP['digammaBoth'].shape:
    LP['digammaBoth'] = np.empty(shp)
    LP['E_logVd'] = np.empty(shp)
    LP['E_log1-Vd'] = np.empty(shp)

  np.add(LP['U0'], LP['U1'], out=LP['digammaBoth'])    
  if activeDocs is None or activeDocs.size > 0.75 * shp[0]:
    digamma(LP['digammaBoth'], out=LP['digammaBoth'])  
    digamma(LP['U0'], out=LP['E_log1-Vd'])
    digamma(LP['U1'], out=LP['E_logVd'])
    LP['E_log1-Vd'] -= LP['digammaBoth']
    LP['E_logVd'] -= LP['digammaBoth']
  else:
    # Fast, optimized version (allocates small memory)
    dBoth = LP['digammaBoth'].take(activeDocs, axis=0)
    digamma(dBoth, out=dBoth)
    d1 = LP['U1'].take(activeDocs,axis=0)
    digamma(d1, out=d1)
    LP['E_logVd'][activeDocs] = d1 - dBoth
    digamma(LP['U0'].take(activeDocs,axis=0), out=d1)
    LP['E_log1-Vd'][activeDocs] = d1 - dBoth
    # Slower, but still decent version
    #LP['digammaBoth'][activeDocs] = digamma(LP['digammaBoth'][activeDocs])
    #LP['E_log1-Vd'][activeDocs] = digamma(LP['U0'][activeDocs])
    #LP['E_logVd'][activeDocs] = digamma(LP['U1'][activeDocs])
    #LP['E_log1-Vd'][activeDocs] -= LP['digammaBoth'].take(activeDocs,axis=0)
    #LP['E_logVd'][activeDocs] -= LP['digammaBoth'].take(activeDocs,axis=0)

  LP['E_logPi'] = LP['E_logVd'].copy()
  LP['E_logPi'][:, 1:] += np.cumsum(LP['E_log1-Vd'][:,:-1], axis=1)
  return LP

def calcDocTopicRemCount(Ndk, out=None):
  ''' Given doc-topic counts, compute "remaining mass" beyond each topic.

     Returns
     --------
     Rdk : 2D array, size nDoc x K
           Rdk[d, k] = \sum_{m=k+1}^K Ndk[d,m]

     Examples
     --------
     >>> calcDocTopicRemCount(np.eye(3))
     [0 0 0]
     [1 0 0]
     [1 1 0]
  '''
  shape = (Ndk.shape[0], Ndk.shape[1])
  if out is None or out.shape != shape:
    out = np.zeros(shape)
  else:
    out[:,0].fill(0)
  np.cumsum(np.fliplr(Ndk[:, 1:]), axis=1, out=out[:,1:])
  return np.fliplr(out)
