''' LocalStep routines, for StickBreak (SB) version of the HDP
'''

import numpy as np
from scipy.special import digamma, gammaln

from bnpy.util import NumericUtil, LibLocalStep

########################################################### doc-level beta
###########################################################  version
def calcLocalDocParams(Data, LP, topicPrior1, topicPrior0, 
                             nCoordAscentItersLP=20,
                             convThrLP=0.01,
                             methodLP='numpy',
                             doUniformFirstTime=False, 
                             **kwargs):
  ''' Calculate local paramters for all documents, given topic prior

      Args
      -------

      Returns 
      -------
      LP : dictionary with fields
  '''
  D = Data.nDoc
  K = topicPrior1.size

  # Precompute ONCE exp( E_logsoftev ), in-place
  expEloglik = LP['E_logsoftev_WordsData']
  expEloglik -= expEloglik.max(axis=1)[:,np.newaxis] 
  NumericUtil.inplaceExp(expEloglik)  
  if methodLP != 'numpy':
    if not np.isfortran(expEloglik):
      expEloglik = np.asfortranarray(expEloglik)


  ######## Allocate document-specific variables
  docptr = np.asarray(np.hstack([0, Data.doc_range[:,1]]), dtype=np.int32)
  if 'DocTopicCount' in LP:
    doUniformFirstTime = False
    # Update U1, U0
    LP = update_U1U0_SB(LP, topicPrior1, topicPrior0)
    # Update expected value of log Pi[d,k]
    LP = update_ElogPi_SB(LP)
    if methodLP == 'numpy':
      expElogpi = np.exp(LP['E_logPi'])
    else:
      expElogpi = np.empty((D,K), order='F')
      np.exp(LP['E_logPi'], out=expElogpi)
  else:
    if methodLP == 'numpy':
      LP['DocTopicCount'] = np.zeros((D, K))
      expElogpi = np.ones((D,K))
    else:
      LP['DocTopicCount'] = np.zeros((D, K), order='F')
      expElogpi = np.ones((D,K), order='F')
    doUniformFirstTime = True

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
    LP = update_U1U0_SB(LP, topicPrior1, topicPrior0)

    # Update expected value of log Pi[d,k]
    LP = update_ElogPi_SB(LP, activeDocs)
    
    # Assess convergence
    docDiffs = np.max(np.abs(old_DocTopicCount - LP['DocTopicCount']), axis=1)
    if np.max(docDiffs) < convThrLP:
      break
    activeDocs = np.int32(np.flatnonzero(docDiffs > convThrLP))

    # Store DocTopicCount for next round's convergence test
    # Here, the "[:]" syntax ensures we do NOT copy the pointer
    old_DocTopicCount[:] = LP['DocTopicCount']
    ### end loop over alternating-ascent updates

  LP['didConverge'] = np.max(docDiffs) < convThrLP
  LP['maxDocDiff'] = np.max(docDiffs)
  LP['nCoordAscentIters'] = ii
  LP['sumRTilde'] = sumRTilde
  LP['expElogpi'] = expElogpi
  LP['expEloglik'] = expEloglik
  del LP['E_logsoftev_WordsData']
  return LP



########################################################### doc-level beta
########################################################### helpers
def update_U1U0_SB(LP, topicPrior1, topicPrior0):
  ''' Update document-level stick-breaking beta parameters, U1 and U0.
  '''
  assert 'DocTopicCount' in LP
  K =  LP['DocTopicCount'].shape[1]
  if 'U1' not in LP:
    LP['U1'] = LP['DocTopicCount'] + topicPrior1
    LP['U0'] = calcDocTopicRemCount(LP['DocTopicCount']) + topicPrior0
  else:
    # no new memory allocated here
    LP['U1'][:] = LP['DocTopicCount'] + topicPrior1
    calcDocTopicRemCount(LP['DocTopicCount'], out=LP['U0'][:,::-1])
    LP['U0'] += topicPrior0
  return LP

def update_ElogPi_SB(LP, activeDocs=None):
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
