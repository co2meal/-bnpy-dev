''' CandidateSelection.py

Selects single topic to consider with a delete move
'''
import numpy as np
import DeleteLogger

def selectCandidateTopic(SS, Data, preselectroutine=None,
                                   randstate=np.random.RandomState(0),
                                   **kwargs):
  ''' 
      Returns
      --------
      ktarget : int ID of a topic to delete
      neighbors : list of ints
                  each entry is ID of topic that ktarget is pos correlated with
  '''
  Info = dict()
  # Verify input args satisfactory
  if SS is None or not SS.hasSelectionTerms():
    Info['msg'] = 'SKIPPED. SuffStatBag needs Selection terms.'
    DeleteLogger.log(Info['msg'])
    return Info
  
  K = SS.K
  D = SS.nDoc
  Smat = SS.getSelectionTerm('DocTopicPairMat')
  svec = SS.getSelectionTerm('DocTopicSum')

  # Remove NaN entries and topics with very little mass from consideration
  topicMinSize = kwargs['topicMinSize']
  offlimitcompIDs = np.flatnonzero(np.logical_or(np.isnan(svec), 
                                                 svec < topicMinSize))
  nanIDs = np.isnan(Smat)
  Smat[nanIDs] = 0
  svec[np.isnan(svec)] = 0

  # Make a proper Cov and Corr matrix
  CovMat = Smat / D - np.outer(svec/D, svec/D)
  varc = np.diag(CovMat)
  assert varc.min() >= 0
  sqrtc = np.sqrt(varc)
  sqrtc[offlimitcompIDs] = 1e-20
  CorrMat = CovMat / np.outer(sqrtc, sqrtc)

  # Now, filter to keep only the *positive* entries in upper diagonal
  #  we shouldn't even bother trying to merge topics with neg correlations
  CorrMat[np.tril_indices(K)] = 0
  CorrMat[CorrMat < 1e-2] = 0
  for kk in offlimitcompIDs:
    CorrMat[kk,:] = 0
    CorrMat[:, kk] = 0

  ps = CorrMat.flatten() 
  Info['CorrMat'] = CorrMat
  Info['ps'] = ps

  # Check if any candidates exist.
  if np.sum(ps) < 1e-9:
    Info['msg'] = 'SKIPPED. No topic pair has positive correlation.'
    DeleteLogger.log(Info['msg'])
    return Info

  ps = ps / np.sum(ps)
  flatID = randstate.choice(CorrMat.size, 1, p=ps)
  rowID, colID = np.unravel_index(flatID, CorrMat.shape)
  rowID = int(rowID)
  colID = int(colID)

  candidateTopics = [rowID, colID]
  for k in xrange(K):
    if k == rowID or k == colID:
      continue
    rCorr = CorrMat[np.minimum(k,rowID),
                       np.maximum(k,rowID)]
    cCorr = CorrMat[np.minimum(k,colID),
                       np.maximum(k,colID)]
    if rCorr > 0 and cCorr > 0:
      candidateTopics.append(k)
  candidateTopics = [x for x in sorted(candidateTopics)]

  topicSizes = SS.N[candidateTopics]
  targetID = np.argmin(topicSizes)
  ktarget = candidateTopics[targetID]

  for posID, kk in enumerate(candidateTopics):
    if posID == 0:
      msg = '  %3d  N=%6.0f  Corr=%.2f' % (kk, SS.N[kk], 
                                  CorrMat[kk, candidateTopics[posID+1]])
    else:
      msg = '  %3d  N=%6.0f  Corr=%.2f' % (kk, SS.N[kk],
                                  CorrMat[candidateTopics[posID-1],kk])
    DeleteLogger.log(msg)    

  DeleteLogger.log('Selected: %d' % (ktarget))
  
  if 'doVizDelete' in kwargs and kwargs['doVizDelete']:
    from matplotlib import pylab
    pylab.imshow( SS.WordCounts[candidateTopics,:], 
                aspect=900./len(candidateTopics), 
                vmax=100, interpolation='nearest')
    pylab.draw()
    pylab.show(block=0)
  
  Info['ktarget'] = ktarget
  Info['neighbors'] = [x for x in candidateTopics if x is not ktarget]
  return Info
