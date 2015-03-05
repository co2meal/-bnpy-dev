import numpy as np
import copy
from scipy.special import digamma, gammaln

from bnpy.util import NumericUtil
import LocalStepLogger

import LocalStepSingleDoc as SDL
import multiprocessing
from multiprocessing import Process
import multiprocessing.sharedctypes
import ctypes

def calcLocalParams(Data, LP, aModel, 
                          methodLP='scratch',
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

  ## Use DocTopicCount matrix (D x K) if passed in
  K = Lik.shape[1]
  hasDocTopicCount = 'DocTopicCount' in LP \
                     and LP['DocTopicCount'].shape == (Data.nDoc, K)
  if methodLP == 'memo' and hasDocTopicCount:
    initDocTopicCount = LP['DocTopicCount'].copy()
  else:
    initDocTopicCount = None

  ## Loop over all documents & do inference!
  DocTopicCount, Prior, sumR, AI = inferLocalParamsForBatch(
                                    Data, aModel, Lik, 
                                    initDocTopicCount, **kwargs)

  LP['DocTopicCount'] = DocTopicCount
  LP = aModel.updateLPGivenDocTopicCount(LP, DocTopicCount)
  LP = updateLPWithResp(LP, Data, Lik, Prior, sumR)

  if 'lapFrac' in kwargs and 'batchID' in kwargs:
    if hasattr(Data, 'batchID') and Data.batchID == kwargs['batchID']:     
      perc = [0, 5, 10, 50, 90, 95, 100]
      siter = ' '.join(['%4d' % np.percentile(AI['iter'], p) for p in perc])
      sdiff = ['%6.4f' % np.percentile(AI['maxDiff'], p) for p in perc]
      sdiff = ' '.join(sdiff)
      nFail = np.sum(AI['maxDiff'] > kwargs['convThrLP'])
      msg = '%4.2f %3d %4d %s %s' % (kwargs['lapFrac'], Data.batchID,
                                     nFail, siter, sdiff)
      worstDocID = np.argmax(AI['maxDiff'])
      msg += " %4d" % (worstDocID)
      if kwargs['restartremovejunkLP'] == 1:
        msg += " %4d/%4d %4d/%4d" % (RInfo['nDocRestartsAccepted'],
                                     RInfo['nDocRestartsTried'],
                                     RInfo['nRestartsAccepted'],
                                     RInfo['nRestartsTried'])
      elif kwargs['restartremovejunkLP'] == 2:
        msg += " %4d/%4d" % (AI['nRestartsAccepted'],
                             AI['nRestartsTried'])
      LocalStepLogger.log(msg)
  LP['Info'] = AI
  return LP

#Method that I wrote
#Fake fake the one I am altering
def inferLocalParamsForBatch(Data, aModel, Lik,
                   initDocTopicCount=None,
                   **kwargs
                  ):
  nProcesses=7
  Processes=[]

  #TODO: change!!! This doesn't fix anything (with regards to nDoc)!!
  try:
    nDoc=Data.nDoc
  except Exception:
    nDoc=2000

  #Note: for this method had to make all of these 1-d arrays
  doc_range=Data.doc_range
  N, K = Lik.shape
  alpha=0.5
  sumRespTilde = np.zeros(N) 
  ## Initialize DocTopicCount and Prior
  DocTopicCount =  np.zeros(nDoc*K)
  Prior = np.zeros(nDoc*K)

  if str(type(aModel)).count('HDP'):
    alphaEbeta = aModel.alphaEbeta[:-1].copy()
    alphaEbetaRem = aModel.alphaEbeta[-1] * 1.0 # to float
  elif str(type(aModel)).count('Finite'):
    alphaEbeta = alpha * np.ones(K)
    alphaEbetaRem = None

  AggInfo = dict()
  AggInfoMaxDiff = np.zeros(nDoc) 
  AggInfoIter = np.zeros(nDoc, dtype=np.int32)

  DocTopicCount=multiprocessing.Array(ctypes.c_double,DocTopicCount, lock=False) 
  Prior= multiprocessing.Array(ctypes.c_double,Prior,lock=False)  
  sumRespTilde=multiprocessing.Array(ctypes.c_double,sumRespTilde,lock=False) 
  AggInfoMaxDiff=multiprocessing.Array(ctypes.c_double,AggInfoMaxDiff,lock=False) 
  AggInfoIter=multiprocessing.Array(ctypes.c_double,AggInfoIter,lock=False) 

  if hasattr(Data, 'word_count'):
    Datawordcount= Data.word_count
  else:
    Datawordcount=None

  dx=nDoc/nProcesses
  start=0
  end=0
  while end<=nDoc: 
    end+=dx
    p=Process(target=tasked,args=(start,min(end,nDoc),doc_range,Lik, DocTopicCount, Prior, sumRespTilde, AggInfoMaxDiff, AggInfoIter,K,alphaEbeta,initDocTopicCount,Datawordcount)) 
    Processes.append(p)
    p.start()
    start+=dx
  
  for p in Processes:
    p.join()

  DocTopicCount=np.array(list(DocTopicCount))
  DocTopicCount=np.reshape(DocTopicCount,(nDoc,K))
  Prior=np.array(list(Prior))
  Prior=np.reshape(Prior,(nDoc,K))
  sumRespTilde=np.array(list(sumRespTilde))
  AggInfo['maxDiff']=list(AggInfoMaxDiff)
  AggInfo['iter']=list(AggInfoIter)
  return DocTopicCount, Prior, sumRespTilde, AggInfo

def tasked(begin,end,doc_range,Lik,DocTopicCount, Prior, sumRespTilde, AggInfoMaxDiff, AggInfoIter,  K,alphaEbeta,initDocTopicCount,Datawordcount,**kwargs):

  for d in range(begin,end):
    # if d%10==0:
    #   print d

    try:
      initDTC_d = initDocTopicCount[d]
    except Exception:
      initDTC_d = None

    start = doc_range[d]
    stop  = doc_range[d+1]
    Lik_d = Lik[start:stop].copy() # Local copy

    if Datawordcount!=None:
      wc_d = Datawordcount[start:stop].copy()
    else:
      wc_d = 1.0
    alphaEbetaRem= None #only works for finite not others!

    #TODO: thought it should be -1 on second param but worked without!!
    DocTopicCount[d*K:d*K+K], Prior[d*K:d*K+K], sumR_d, Info = SDL.inferLocal_SingleDoc_Dir(
                                        wc_d, Lik_d, 
                                        alphaEbeta, alphaEbetaRem,
                                        initDTC_d,
                                        **kwargs)

    sumRespTilde[int(start):int(stop)] = sumR_d

    ## Fill in AggInfo fields

    AggInfoMaxDiff[d] = Info['maxDiff']
    AggInfoIter[d] = Info['iter']

    #Had to comment out the below ones, since AggInfo isn't a dictionary

    #TODO: it looks as though ELBOtrace is sequential in that it is overwritten by later docs (this would not work otherwise)
    # if 'ELBOtrace' in Info:
    #   AggInfo['ELBOtrace'] = Info['ELBOtrace']
    # if 'nAccept' in Info:
    #   if 'nRestartsAccepted' not in AggInfo:
    #     AggInfo['nRestartsAccepted'] = 0
    #     AggInfo['nRestartsTried'] = 0
    #   AggInfo['nRestartsAccepted'] += Info['nAccept']
    #   AggInfo['nRestartsTried'] += Info['nTrial']


#Below is the real method
def inferLocalParamsForBatchReal(Data, aModel, Lik,
                   initDocTopicCount=None,
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
  N, K = Lik.shape
  sumRespTilde = np.zeros(N)
  ## Initialize DocTopicCount and Prior
  DocTopicCount = np.zeros((Data.nDoc, K))
  Prior = np.zeros((Data.nDoc, K))

  if str(type(aModel)).count('HDP'):
    alphaEbeta = aModel.alphaEbeta[:-1].copy()
    alphaEbetaRem = aModel.alphaEbeta[-1] * 1.0 # to float
  elif str(type(aModel)).count('Finite'):
    alphaEbeta = aModel.alpha * np.ones(K)
    alphaEbetaRem = None

  AggInfo = dict()
  AggInfo['maxDiff'] = np.zeros(Data.nDoc)
  AggInfo['iter'] = np.zeros(Data.nDoc, dtype=np.int32)
  for d in xrange(Data.nDoc):
    start = Data.doc_range[d]
    stop  = Data.doc_range[d+1]
    Lik_d = Lik[start:stop].copy() # Local copy
    if hasattr(Data, 'word_count'):
      wc_d = Data.word_count[start:stop].copy()
    else:
      wc_d = 1.0

    try:
      initDTC_d = initDocTopicCount[d]
    except Exception:
      initDTC_d = None

    DocTopicCount[d], Prior[d], sumR_d, Info = SDL.inferLocal_SingleDoc_Dir(
                                      wc_d, Lik_d, 
                                      alphaEbeta, alphaEbetaRem,
                                      initDTC_d,
                                      **kwargs)
    sumRespTilde[start:stop] = sumR_d

    ## Fill in AggInfo fields
    AggInfo['maxDiff'][d] = Info['maxDiff']
    AggInfo['iter'][d] = Info['iter']
    if 'ELBOtrace' in Info:
      AggInfo['ELBOtrace'] = Info['ELBOtrace']
    if 'nAccept' in Info:
      if 'nRestartsAccepted' not in AggInfo:
        AggInfo['nRestartsAccepted'] = 0
        AggInfo['nRestartsTried'] = 0
      AggInfo['nRestartsAccepted'] += Info['nAccept']
      AggInfo['nRestartsTried'] += Info['nTrial']
  return DocTopicCount, Prior, sumRespTilde, AggInfo


def updateLPWithResp(LP, Data, Lik, Prior, sumRespTilde):
  LP['resp'] = Lik.copy()
  for d in xrange(Data.nDoc):
    start = Data.doc_range[d]
    stop  = Data.doc_range[d+1]
    LP['resp'][start:stop] *= Prior[d]
  LP['resp'] /= sumRespTilde[:, np.newaxis]
  np.maximum(LP['resp'], 1e-300, out=LP['resp'])
  return LP

def updateSingleDocLPWithResp(LP_d, Lik_d, Prior_d, sumR_d):
  resp_d = Lik_d.copy()
  resp_d *= Prior_d
  resp_d /= sumR_d[:, np.newaxis]
  np.maximum(resp_d, 1e-300, out=resp_d)
  LP_d['resp'] = resp_d
  return LP_d
