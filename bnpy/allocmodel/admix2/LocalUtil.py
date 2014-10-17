import numpy as np
from bnpy.util import NumericUtil

nCoordAscentIters = 20
convThr = 0.001

def calcLocalParams(Data, LP, aModel, 
                          methodLP='scratch',
                          routineLP='simple',
                          **kwargs):
  ''' Calculate all local parameters for provided dataset under a topic model

      Returns
      -------
      LP : dict of local params, with fields
      * DocTopicCount
      * resp
      * model-specific fields for doc-topic probabilities
  ''' 
  kwargs['methodLP'] = methodLP

  ## Prepare the log soft ev matrix
  ## Make sure it is C-contiguous, so that matrix ops are very fast
  Lik = np.asarray(LP['E_log_soft_ev'], order='C') 
  Lik -= Lik.max(axis=1)[:,np.newaxis] 
  NumericUtil.inplaceExp(Lik)

  K = Lik.shape[1]
  hasDocTopicCount = 'DocTopicCount' in LP \
                     and LP['DocTopicCount'].shape == (Data.nDoc, K)
  if methodLP == 'memo' and hasDocTopicCount:
    initDocTopicCount = LP['DocTopicCount']
  else:
    initDocTopicCount = None

  if routineLP == 'simple':
    DocTopicCount, Prior, sumR = calcDocTopicCountForData_Simple(Data, aModel,
                                      Lik,
                                      initDocTopicCount=initDocTopicCount,
                                      **kwargs)
  elif routineLP == 'fast':
    DocTopicCount, Prior, sumR = calcDocTopicCountForData_Fast(Data, aModel,
                                      Lik,
                                      initDocTopicCount=initDocTopicCount,
                                      **kwargs)
  else:
    raise ValueError('Unrecognized routine ' + routineLP)

  LP = aModel.updateLPGivenDocTopicCount(LP, DocTopicCount)
  LP = updateLPWithResp(LP, Data, Lik, Prior, sumR)
  LP['DocTopicCount'] = DocTopicCount
  return LP
  

def updateLPWithResp(LP, Data, Lik, Prior, sumRespTilde):
  LP['resp'] = Lik.copy()
  for d in xrange(Data.nDoc):
    start = Data.doc_range[d]
    stop  = Data.doc_range[d+1]
    LP['resp'][start:stop] *= Prior[d]
  LP['resp'] /= sumRespTilde[:, np.newaxis]
  np.maximum(LP['resp'], 1e-300, out=LP['resp'])
  return LP

def calcDocTopicCountForData_Simple(Data, aModel, Lik,
                   initDocTopicCount=None,
                   initPrior=None, 
                   **kwargs
                  ):
  ''' Calculate updated doc-topic counts for every document in provided set

      Will loop over all docs, and at each one will run coordinate ascent
      to alternatively update its doc-topic counts and the doc-topic prior.
      Ascent stops after convergence or a maximum number of iterations.
    
      Returns
      ---------
      DocTopicCount : 2D array, size nDoc x K
      DocTopicCount[d,k] is effective number of tokens in doc d assigned to k

      Prior : 2D array, size nDoc x K
      Prior[d,k] = exp( E[log pi_{dk}] )

      sumRespTilde : 1D array, size N = # observed tokens
                     sumRespTilde[n] = normalization for the responsibility          
                     parameters for token n
  '''
  sumRespTilde = np.zeros(Lik.shape[0])

  ## Initialize DocTopicCount and Prior
  if initDocTopicCount is not None:
    DocTopicCount = initDocTopicCount.copy()
    Prior = aModel.calcLogPrActiveComps_Fast(DocTopicCount)
    Prior -= Prior.max(axis=1)[:, np.newaxis]
    np.exp(Prior, out=Prior)
  else:
    DocTopicCount = np.zeros((Data.nDoc, aModel.K))
    if initPrior is None:
      if kwargs['methodLP'] == 'scratch':
        Prior = np.ones((Data.nDoc, aModel.K))
      elif kwargs['methodLP'] == 'prior':
        Prior = np.tile(aModel.get_active_comp_probs(), (Data.nDoc, 1))
      else:
        Prior = np.ones((Data.nDoc, aModel.K))
    else:
      Prior = initPrior.copy()

  for d in xrange(Data.nDoc):
    start = Data.doc_range[d]
    stop  = Data.doc_range[d+1]
    Lik_d = Lik[start:stop].copy() # Local copy
    if hasattr(Data, 'word_count'):
      wc_d = Data.word_count[start:stop].copy()
    else:
      wc_d = 1.0
    sumR_d = np.zeros(stop-start)

    DocTopicCount[d], Prior[d], sumR_d = calcDocTopicCountForDoc(
                                      d, aModel, 
                                      DocTopicCount[d], Lik_d,
                                      Prior[d], sumR_d, 
                                      wc_d,
                                      **kwargs)

    sumRespTilde[start:stop] = sumR_d

  return DocTopicCount, Prior, sumRespTilde


def calcDocTopicCountForDoc(d, aModel,
                            DocTopicCount_d, Lik_d,
                            Prior_d, sumR_d, 
                            wc_d=None,
                            nCoordAscentItersLP=nCoordAscentIters,
                            convThrLP=convThr,
                            **kwargs
                            ):
  '''
     Returns
      ---------
      DocTopicCount : 1D array, size K
                      DocTopicCount[k] is effective number of tokens 
                      assigned to topic k in the current document d

      Prior_d      : 1D array, size K
                     Prior_d[k] : probability of topic k in current doc d

      sumRespTilde : 1D array, size Nd = # observed tokens in current doc d
                     sumRespTilde[n] = normalization for the responsibility          
                     parameters for token n
  '''
  prevDocTopicCount_d = DocTopicCount_d.copy()
  if hasattr(aModel, 'calcLogPrActiveCompsForDoc'):
    aFunc = aModel.calcLogPrActiveCompsForDoc
  else:
    aFunc = aModel

  for iter in xrange(nCoordAscentItersLP):
    ## Update Prob of Active Topics
    if iter > 0:
      aFunc(DocTopicCount_d, Prior_d) # Prior_d = E[ log pi_dk ]
      np.exp(Prior_d, out=Prior_d)    # Prior_d = exp E[ log pi_dk ]

    ## Update sumR_d for all tokens in document
    np.dot(Lik_d, Prior_d, out=sumR_d)

    ## Update DocTopicCounts
    np.dot(wc_d / sumR_d, Lik_d, out=DocTopicCount_d)
    DocTopicCount_d *= Prior_d

    ## Check for convergence
    maxDiff = np.max(np.abs(DocTopicCount_d - prevDocTopicCount_d))
    if maxDiff < convThrLP:
      break
    prevDocTopicCount_d[:] = DocTopicCount_d

  return DocTopicCount_d, Prior_d, sumR_d



def calcDocTopicCountForData_Fast(Data, *args, **kwargs):
  if hasattr(Data, 'word_count'):
    return calcDocTopicCountForData_Fast_wordcount(Data, *args, **kwargs)
  else:
    return calcDocTopicCountForData_Fast_nowordcount(Data, *args, **kwargs)

def calcDocTopicCountForData_Fast_wordcount(Data, aModel, Lik,
                   initDocTopicCount=None,
                   initPrior=None, 
                   nCoordAscentItersLP=nCoordAscentIters,
                   convThrLP=convThr,
                   **kwargs
                  ):
  ''' Calculate updated doc-topic counts for every document in provided set

      Will loop over all docs, and at each one will run coordinate ascent
      to alternatively update its doc-topic counts and the doc-topic prior.
      Ascent stops after convergence or a maximum number of iterations.
    
      Returns
      ---------
      DocTopicCount : 2D array, size nDoc x K
      DocTopicCount[d,k] is effective number of tokens in doc d assigned to k

      Prior : 2D array, size nDoc x K
      Prior[d,k] = exp( E[log pi_{dk}] )

      sumRespTilde : 1D array, size N = # observed tokens
                     sumRespTilde[n] = normalization for the responsibility          
                     parameters for token n
  '''
  ## Initialize 
  tmpLP = dict()
  sumRespTilde = np.zeros(Lik.shape[0])

  if initDocTopicCount is not None:
    DocTopicCount = initDocTopicCount.copy()
    Prior = aModel.calcLogPrActiveComps_Fast(DocTopicCount, None, tmpLP)
    np.exp(Prior, out=Prior)
  else:
    DocTopicCount = np.zeros((Data.nDoc, aModel.K))
    if initPrior is None:
      Prior = np.ones((Data.nDoc, aModel.K))
    else:
      Prior = initPrior.copy()

  activeDocs = np.arange(Data.nDoc, dtype=np.int32)
  prev_DocTopicCount = DocTopicCount.copy()

  for ii in xrange(nCoordAscentItersLP):
    ## Update Prior for active documents
    if ii > 0:
      aModel.calcLogPrActiveComps_Fast(DocTopicCount, activeDocs, tmpLP,
                                       out=Prior)
      # Unfortunately, cannot update only activeDocs inplace (fancy idxing)
      Prior[activeDocs] = np.exp(Prior[activeDocs])

    for d in activeDocs:
      start = Data.doc_range[d]
      stop = Data.doc_range[d+1]
      Lik_d = Lik[start:stop]

      ## Update sumRtilde for all tokens in document
      np.dot(Lik_d, Prior[d], out=sumRespTilde[start:stop])

      ## Update DocTopicCount with Likelihood
      wc_d = Data.word_count[start:stop]
      np.dot(wc_d / sumRespTilde[start:stop], Lik_d, out=DocTopicCount[d])
 
    ## Update DocTopicCount with Prior
    DocTopicCount[activeDocs] *= Prior[activeDocs]

    # Assess convergence
    docDiffs = np.max(np.abs(prev_DocTopicCount - DocTopicCount), axis=1)
    if np.max(docDiffs) < convThrLP:
      break
    activeDocs = np.asarray(np.flatnonzero(docDiffs >= convThrLP),
                            dtype=np.int32)

    # Store DocTopicCount for next round's convergence test
    # Here, the "[:]" syntax ensures we do NOT copy by reference
    prev_DocTopicCount[activeDocs] = DocTopicCount[activeDocs]
    ### end loop over alternating-ascent updates

  return DocTopicCount, Prior, sumRespTilde



def calcDocTopicCountForData_Fast_nowordcount(Data, aModel, Lik,
                   initDocTopicCount=None,
                   initPrior=None, 
                   nCoordAscentItersLP=nCoordAscentIters,
                   convThrLP=convThr,
                   **kwargs
                  ):
  ''' Calculate updated doc-topic counts for every document in provided set

      Will loop over all docs, and at each one will run coordinate ascent
      to alternatively update its doc-topic counts and the doc-topic prior.
      Ascent stops after convergence or a maximum number of iterations.
    
      Returns
      ---------
      DocTopicCount : 2D array, size nDoc x K
      DocTopicCount[d,k] is effective number of tokens in doc d assigned to k

      Prior : 2D array, size nDoc x K
      Prior[d,k] = exp( E[log pi_{dk}] )

      sumRespTilde : 1D array, size N = # observed tokens
                     sumRespTilde[n] = normalization for the responsibility          
                     parameters for token n
  '''
  ## Initialize 
  tmpLP = dict()
  sumRespTilde = np.zeros(Lik.shape[0])

  if initDocTopicCount is not None:
    DocTopicCount = initDocTopicCount.copy()
    Prior = aModel.calcLogPrActiveComps_Fast(DocTopicCount, None, tmpLP)
    np.exp(Prior, out=Prior)
  else:
    DocTopicCount = np.zeros((Data.nDoc, aModel.K))
    if initPrior is None:
      Prior = np.ones((Data.nDoc, aModel.K))
    else:
      Prior = initPrior.copy()

  activeDocs = np.arange(Data.nDoc, dtype=np.int32)
  prev_DocTopicCount = DocTopicCount.copy()

  for ii in xrange(nCoordAscentItersLP):
    ## Update Prior for active documents
    if ii > 0:
      aModel.calcLogPrActiveComps_Fast(DocTopicCount, activeDocs, tmpLP,
                                       out=Prior)
      # Unfortunately, cannot update only activeDocs inplace (fancy idxing)
      Prior[activeDocs] = np.exp(Prior[activeDocs])

    for d in activeDocs:
      start = Data.doc_range[d]
      stop = Data.doc_range[d+1]
      Lik_d = Lik[start:stop]

      ## Update sumRtilde for all tokens in document
      np.dot(Lik_d, Prior[d], out=sumRespTilde[start:stop])

      ## Update DocTopicCount with Likelihood
      np.dot(1.0 / sumRespTilde[start:stop], Lik_d, out=DocTopicCount[d])
    ## Update DocTopicCount with Prior
    DocTopicCount[activeDocs] *= Prior[activeDocs]

    # Assess convergence
    docDiffs = np.max(np.abs(prev_DocTopicCount - DocTopicCount), axis=1)
    if np.max(docDiffs) < convThrLP:
      break
    activeDocs = np.asarray(np.flatnonzero(docDiffs >= convThrLP),
                            dtype=np.int32)

    # Store DocTopicCount for next round's convergence test
    # Here, the "[:]" syntax ensures we do NOT copy by reference
    prev_DocTopicCount[activeDocs] = DocTopicCount[activeDocs]
    ### end loop over alternating-ascent updates

  return DocTopicCount, Prior, sumRespTilde


def printVectors(aname, a, fmt='%9.6f', Kmax=10):
  if len(a) > Kmax:
    print 'FIRST %d' % (Kmax)
    printVectors(aname, a[:Kmax], fmt, Kmax)
    print 'LAST %d' % (Kmax)
    printVectors(aname, a[-Kmax:], fmt, Kmax)

  else:
    print ' %10s %s' % (aname, np2flatstr(a, fmt, Kmax))

def np2flatstr(xvec, fmt='%9.3f', Kmax=10):
  return ' '.join( [fmt % (x) for x in xvec[:Kmax]])

