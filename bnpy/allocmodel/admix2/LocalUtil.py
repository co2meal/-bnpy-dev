import numpy as np
from bnpy.util import NumericUtil

nCoordAscentIters = 20
convThr = 0.001

def calcLocalParams(Data, LP, aModel, methodLP='scratch', **kwargs):
  ''' Calculate all local parameters for provided dataset under a topic model

      Returns
      -------
      LP : dict of local params, with fields
      * DocTopicCount
      * resp
      * model-specific fields for doc-topic probabilities
  ''' 
  ## Prepare the log soft ev matrix 
  Lik = LP['E_log_soft_ev']
  Lik -= Lik.max(axis=1)[:,np.newaxis] 
  NumericUtil.inplaceExp(Lik)

  K = Lik.shape[1]
  hasDocTopicCount = 'DocTopicCount' in LP \
                     and LP['DocTopicCount'].shape == (Data.nDoc, K)
  if methodLP == 'memo' and hasDocTopicCount:
    initDocTopicCount = LP['DocTopicCount']
  else:
    initDocTopicCount = None

  DocTopicCount, Prior, sumR = calcDocTopicCountForData(Data, aModel, Lik,
                                      initDocTopicCount=initDocTopicCount,
                                      **kwargs)

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

def calcDocTopicCountForData(Data, aModel, Lik,
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
  if initDocTopicCount is None:
    DocTopicCount = np.zeros((Data.nDoc, aModel.K))
  else:
    DocTopicCount = initDocTopicCount.copy()
  if initPrior is None:
    Prior = np.ones((Data.nDoc, aModel.K))
  else:
    Prior = initPrior.copy()
  sumRespTilde = np.zeros(Lik.shape[0])

  for d in xrange(Data.nDoc):
    start = Data.doc_range[d]
    stop  = Data.doc_range[d+1]
    Lik_d = Lik[start:stop]
    sumR_d = np.zeros(stop-start)
    if hasattr(Data, 'word_count'):
      wc_d = Data.word_count[start:stop]
    else:
      wc_d = np.ones(stop-start)

    DocTopicCount[d], Prior[d], sumR_d = calcDocTopicCountForDoc(
                                      d, aModel, 
                                      DocTopicCount[d], Lik_d,
                                      Prior[d], sumR_d, 
                                      wc_d,
                                      nCoordAscentItersLP,
                                      convThrLP)

    sumRespTilde[start:stop] = sumR_d

  return DocTopicCount, Prior, sumRespTilde


def calcDocTopicCountForDoc(d, aModel,
                            DocTopicCount_d, Lik_d,
                            Prior_d, sumR_d, 
                            wc_d=None,
                            nCoordAscentItersLP=nCoordAscentIters,
                            convThrLP=convThr):
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
  for iter in xrange(nCoordAscentItersLP):
    #if d == 0 and (iter == 0 or iter % 5 == 0):
    #  if iter == 0:
    #    print '-----'
    #  print ' '.join(['%7.1f' % (x) for x in DocTopicCount_d])
      
    ## Update Prob of Active Topics
    if iter > 0:
      np.exp(aModel.calcLogPrActiveCompsForDoc(DocTopicCount_d), 
               out=Prior_d)

    ## Update sumRtilde for all tokens in document
    np.dot(Lik_d, Prior_d, out=sumR_d)

    ## Update DocTopicCounts
    np.dot(wc_d / sumR_d, Lik_d, out=DocTopicCount_d)
    DocTopicCount_d *= Prior_d

    ## Check for convergence
    docDiff = np.max(np.abs(DocTopicCount_d - prevDocTopicCount_d))
    if docDiff < convThrLP:
      break
    prevDocTopicCount_d[:] = DocTopicCount_d
  return DocTopicCount_d, Prior_d, sumR_d