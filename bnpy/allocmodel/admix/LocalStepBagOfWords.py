'''
LocalStepBagOfWords.py

Optimized numerical routines for computing local parameters for bag-of-words topic models.
'''
import numpy as np
from scipy.special import digamma

from bnpy.util import NumericUtil 

########################################################### dirichlet helpers
###########################################################

def update_DocTopicCount(Data, LP):
  assert 'word_variational' in LP
  if 'DocTopicCount' not in LP:
    K = LP['word_variational'].shape[1]
    LP['DocTopicCount'] = np.zeros((Data.nDoc, K))
  for d in xrange(Data.nDoc):
    start = Data.doc_range[d,0]
    stop = Data.doc_range[d,1]
    LP['DocTopicCount'][d,:] = np.dot(
                                     Data.word_count[start:stop],        
                                     LP['word_variational'][start:stop,:]
                                     )
  return LP

def update_theta(LP, topicPrior, unusedTopicPrior=None):
  assert 'DocTopicCount' in LP
  K =  LP['DocTopicCount'].shape[1]
  if 'theta' not in LP:
    LP['theta'] = LP['DocTopicCount'] + topicPrior[np.newaxis,:]
  else:
    LP['theta'][:] = LP['DocTopicCount'] + topicPrior[np.newaxis,:]
  if unusedTopicPrior is not None:
    LP['theta_u'] = unusedTopicPrior
  return LP

def update_ElogPi(LP, unusedTopicPrior=None):
  ''' Update expected log topic appearance probabilities in each doc
  '''
  if 'E_logPi' in LP and LP['E_logPi'].shape[1] == LP['theta'].shape[1]:
    digamma(LP['theta'], out=LP['E_logPi'])
  else:
    LP['E_logPi'] = digamma(LP['theta'])
  if unusedTopicPrior is None:
    sumTheta = np.sum(LP['theta'], axis=1)
  else:
    sumTheta = np.sum(LP['theta'], axis=1) + unusedTopicPrior
  digammasumTheta = digamma(sumTheta)
  LP['E_logPi'] -= digammasumTheta[:,np.newaxis]
  LP['digammasumTheta'] = digammasumTheta
  if unusedTopicPrior is not None:
    LP['E_logPi_u'] = digamma(unusedTopicPrior) - digammasumTheta
  return LP

########################################################### MAIN
###########################################################

def calcLocalDocParams(Data, LP, topicPrior, **kwargs):
  ''' User-facing function for local step in topic models
  '''
  if 'localmethod' in kwargs and kwargs['localmethod'] == 'forloop':
    return calcLocalDocParams_forloopoverdocs(Data, LP, topicPrior, **kwargs)
  return calcLocalDocParams_vectorized(Data, LP, topicPrior, **kwargs)

########################################################### vectorized version
###########################################################

def calcLocalDocParams_vectorized(Data, LP, topicPrior, 
                             unusedTopicPrior=None,
                             nCoordAscentItersLP=20,
                             convThrLP=0.01,
                             doUniformFirstTime=False, 
                             **kwargs):
  ''' Returns 
      -------
      LP : dictionary with fields
           theta : 2D array, size nDoc x K
           ElogPi : 2D array, size nDoc x K
           unusedElogPi : 1D array, size nDoc
  '''
  K = topicPrior.size

  # Precompute ONCE exp( E_logsoftev ), in-place
  expEloglik = LP['E_logsoftev_WordsData']
  expEloglik -= expEloglik.max(axis=1)[:,np.newaxis] 
  NumericUtil.inplaceExp(expEloglik)  

  ######## Allocate document-specific variables
  LP['DocTopicCount'] = np.zeros((Data.nDoc, K))
  if 'theta' in LP and LP['theta'].shape[1] == K:
    LP = update_ElogPi(LP, unusedTopicPrior)
  else:
    LP['theta'] = np.zeros((Data.nDoc, K))
    doUniformFirstTime = True

  ######## Allocate token-specific variables
  # sumRTilde : nDistinctWords-length vector of reals
  #   row n = \sum_{k} \tilde{r}_{nk}, 
  #             where \tilde{r}_nk = \exp{ Elog[\pi_d] + Elog[\phi_dvk] }
  #   each entry is the "normalizer" for each row of LP['resp']
  sumRTilde = np.zeros(Data.nObs)

  ######## Repeat updates until old_theta has stopped changing ...
  activeDocs = range(Data.nDoc)
  old_theta = LP['theta'].copy()
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

    # Update theta
    LP['theta'] = LP['DocTopicCount'] + topicPrior[np.newaxis,:]

    # Update expected value of log Pi[d,k]
    LP = update_ElogPi(LP, unusedTopicPrior)
    
    # Assess convergence
    docDiffs = np.max(np.abs(old_theta - LP['theta']), axis=1)
    if np.max(docDiffs) < convThrLP:
      break
    activeDocs = np.flatnonzero(docDiffs > convThrLP)

    # Store previous value of theta for next round's convergence test
    # Here, the "[:]" syntax ensures we do NOT copy the pointer
    old_theta[:] = LP['theta']
    ### end loop over alternating-ascent updates
  
  ######## 
  if unusedTopicPrior is not None:
    LP['theta_u'] = unusedTopicPrior
    LP['E_logPi_u'] = digamma(unusedTopicPrior) - LP['digammasumTheta']
  LP['sumRTilde'] = sumRTilde
  LP['expElogpi'] = expElogpi
  LP['expEloglik'] = expEloglik
  del LP['E_logsoftev_WordsData']
  assert 'digammasumTheta' in LP
  return LP


########################################################### forloop version
###########################################################

def calcLocalDocParams_forloopoverdocs(Data, LP, topicPrior, 
                             unusedTopicPrior=None,
                             nCoordAscentItersLP=20,
                             convThrLP=0.01,
                             doUniformFirstTime=False, **kwargs):
  ''' 

      Returns 
      -------
      LP : dictionary with fields
           theta : 2D array, size nDoc x K
           ElogPi : 2D array, size nDoc x K
           unusedElogPi : 1D array, size nDoc
  '''
  K = topicPrior.size

  # Precompute ONCE exp( E_logsoftev ), in-place
  expEloglik = LP['E_logsoftev_WordsData']
  expEloglik -= expEloglik.max(axis=1)[:,np.newaxis] 
  NumericUtil.inplaceExp(expEloglik)  

  ######## Allocate document-specific variables
  LP['DocTopicCount'] = np.zeros((Data.nDoc, K))
  if 'theta' in LP:
    LP = update_ElogPi(LP, unusedTopicPrior)
  else:
    LP['theta'] = np.zeros((Data.nDoc, K))
    LP['E_logPi'] = np.zeros((Data.nDoc, K))
    doUniformFirstTime = True

  expElogpi = np.zeros( (Data.nDoc, K))
  sumRTilde = np.zeros(Data.nObs)

  for d in xrange(Data.nDoc):

    start = Data.doc_range[d,0]
    stop  = Data.doc_range[d,1]
    expEloglik_d = expEloglik[start:stop]

    old_theta = LP['theta'][d,:].copy()
    for ii in xrange(nCoordAscentItersLP):

      # Update expElogpi
      if doUniformFirstTime and ii == 0:
        expElogpi_d = np.ones(K)
      else:
        expElogpi_d = np.exp(LP['E_logPi'][d,:])

      sumRTilde_d = np.dot(expEloglik_d, expElogpi_d)

      LP['DocTopicCount'][d,:] = np.dot( 
                                    Data.word_count[start:stop] / sumRTilde_d,
                                    expEloglik_d
                                       )

      if not (doUniformFirstTime and ii == 0):
        # Element-wise multiply with nDoc x K prior prob matrix
        LP['DocTopicCount'][d,:] *= expElogpi_d

      # Update theta
      LP['theta'][d,:] = LP['DocTopicCount'][d,:] + topicPrior

      # Update expected value of log Pi[d,k]
      if unusedTopicPrior is None:
        digammasumTheta = digamma(np.sum(LP['theta'][d,:]))
      else:
        digammasumTheta = digamma(np.sum(LP['theta'][d,:]) + unusedTopicPrior)
      LP['E_logPi'][d,:] = digamma(LP['theta'][d,:]) - digammasumTheta

      # Assess convergence
      docDiff = np.max(np.abs(old_theta - LP['theta'][d,:]))
      if np.max(docDiff) < convThrLP:
        break

      # Store previous value of theta for next round's convergence test
      # Here, the "[:]" syntax ensures we do NOT copy the pointer
      old_theta[:] = LP['theta'][d,:]
      ### end loop over alternating-ascent updates
  
    expElogpi[d,:] = expElogpi_d
    sumRTilde[start:stop] = sumRTilde_d
    ### end loop over documents

  ######## 
  LP['expElogpi'] = expElogpi
  LP['expEloglik'] = expEloglik
  LP['sumRTilde'] = sumRTilde
  return LP


"""
def backup():
  LP['docIDs'] = docIDs
  LP['word_variational'] = expEloglik
  for d in xrange(Data.nDoc):
    start = Data.doc_range[d,0]
    stop  = Data.doc_range[d,1]
    LP['word_variational'][start:stop] *= expElogpi[d]
  LP['word_variational'] /= SumRTilde[:, np.newaxis]
"""
