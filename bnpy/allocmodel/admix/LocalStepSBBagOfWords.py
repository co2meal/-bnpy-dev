''' LocalStep routines, for StickBreak (SB) version of the HDP
'''

import numpy as np
from scipy.special import digamma, gammaln

from bnpy.util import NumericUtil 

########################################################### doc-level beta
###########################################################  version
def calcLocalDocParams(Data, LP, topicPrior1, topicPrior0, 
                             nCoordAscentItersLP=20,
                             convThrLP=0.01,
                             doUniformFirstTime=False, 
                             **kwargs):
  ''' Calculate local paramters for all documents, given topic prior

      Args
      -------

      Returns 
      -------
      LP : dictionary with fields
  '''
  K = topicPrior1.size

  # Precompute ONCE exp( E_logsoftev ), in-place
  expEloglik = LP['E_logsoftev_WordsData']
  expEloglik -= expEloglik.max(axis=1)[:,np.newaxis] 
  NumericUtil.inplaceExp(expEloglik)  

  ######## Allocate document-specific variables
  if 'DocTopicCount' in LP:
    doUniformFirstTime = False
  else:
    LP['DocTopicCount'] = np.zeros((Data.nDoc, K))
    doUniformFirstTime = True

  ######## Allocate token-specific variables
  # sumRTilde : nDistinctWords-length vector of reals
  #   row n = \sum_{k} \tilde{r}_{nk}, 
  #             where \tilde{r}_nk = \exp{ Elog[\pi_d] + Elog[\phi_dvk] }
  #   each entry is the "normalizer" for each row of LP['resp']
  sumRTilde = np.zeros(Data.nObs)

  ######## Repeat updates until old_theta has stopped changing ...
  activeDocs = range(Data.nDoc)
  old_DocTopicCount = LP['DocTopicCount'].copy()
  for ii in xrange(nCoordAscentItersLP):

    # Update expElogpi for active documents
    if doUniformFirstTime and ii == 0:
      expElogpi = np.ones((Data.nDoc, K))
    else:
      if len(activeDocs) == Data.nDoc:
        expElogpi = np.exp(LP['E_logPi'])
      else:
        expElogpi[activeDocs] = np.exp(LP['E_logPi'][activeDocs])    

    for d in activeDocs:
      start = Data.doc_range[d,0]
      stop  = Data.doc_range[d,1]
      expEloglik_d = expEloglik[start:stop]

      np.dot(expEloglik_d, expElogpi[d], out=sumRTilde[start:stop])

      np.dot(Data.word_count[start:stop] / sumRTilde[start:stop],
               expEloglik_d,
               out=LP['DocTopicCount'][d,:]
            )

    if not (doUniformFirstTime and ii == 0):
      # Element-wise multiply with nDoc x K prior prob matrix
      LP['DocTopicCount'][activeDocs] *= expElogpi[activeDocs]

    # Update U1, U0
    LP = update_U1U0_SB(LP, topicPrior1, topicPrior0)

    # Update expected value of log Pi[d,k]
    LP = update_ElogPi_SB(LP)
    
    # Assess convergence
    docDiffs = np.max(np.abs(old_DocTopicCount - LP['DocTopicCount']), axis=1)
    if np.max(docDiffs) < convThrLP:
      break
    activeDocs = np.flatnonzero(docDiffs > convThrLP)

    # Store DocTopicCount for next round's convergence test
    # Here, the "[:]" syntax ensures we do NOT copy the pointer
    old_DocTopicCount[:] = LP['DocTopicCount']
    ### end loop over alternating-ascent updates
  
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

def update_ElogPi_SB(LP):
  ''' Update expected log topic appearance probabilities in each doc
  '''
  shape = LP['U1'].shape 
  digammaBoth = digamma(LP['U0']+LP['U1'])
  LP['E_logVd'] = digamma(LP['U1']) - digammaBoth
  LP['E_log1-Vd'] = digamma(LP['U0']) - digammaBoth

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