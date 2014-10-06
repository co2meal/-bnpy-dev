'''
FromTruth.py

Initialize params of a bnpy model using "ground truth" information,
such as human annotations 

These are provided within a Data object, in either
* TrueLabels dict, containing either hard or soft local parameter assignments
* TrueParams dict, containing point-estimates of global parameters

'''
import numpy as np
import FromScratchMult

def init_global_params(hmodel, Data, initname=None, seed=0, **kwargs):
  ''' Initialize global params of given hmodel using Data's ground truth.
      
      Args
      -------
      hmodel : bnpy model object to initialize
      Data   : bnpy Data object whose dimensions must match resulting hmodel
      initname : string name for the routine to use
                 'truelabels' or 'repeattruelabels'

      Returns
      --------
      None. hmodel global params updated in-place.
  '''
  PRNG = np.random.RandomState(seed)
  if initname.count('truelabels') > 0:
    _initFromTrueLP(hmodel, Data, initname, PRNG, **kwargs)
  else:
    _initFromTrueParams(hmodel, Data, initname, PRNG, **kwargs)

  if hmodel.obsModel.inferType == 'EM':
    assert hasattr(hmodel.obsModel, 'EstParams')
  else:
    assert hasattr(hmodel.obsModel, 'Post')


def _initFromTrueParams(hmodel, Data, initname, PRNG, **kwargs):
  ''' Initialize global parameters of provided model to specific values

      Relies on the set_global_params method implemented by all alloc/obs models

      Returns
      --------
      None. hmodel updated in-place.
  '''
  if initname != 'trueparams':
    raise NotImplementedError('Unknown initname: %s' % (initname))
  InitParams = dict(**Data.TrueParams)
  InitParams['Data'] = Data
  hmodel.set_global_params(**InitParams)


def _initFromTrueLP(hmodel, Data, initname, PRNG, nRepeatTrue=2, **kwargs):
  ''' Initialize global parameters of provided model given local parameters

      Relies on update_global_params to set the global params given locals LP

      Returns
      --------
      None. hmodel updated in-place.
  '''

  ## Extract "true" local params dictionary LP specified in the Data struct
  LP = dict()
  if hasattr(Data, 'TrueParams') and 'Z' in Data.TrueParams:
    LP['Z'] = Data.TrueParams['Z']
    LP = convertLPFromHardToSoft(LP, Data)
  elif hasattr(Data, 'TrueParams') and 'resp' in Data.TrueParams:
    LP['resp'] = Data.TrueParams['resp']
  else:
    raise ValueError('init_global_params requires TrueLabels or TrueParams.')

  ## Convert between token/doc responsibilities
  if str(type(hmodel.obsModel)).count('Mult'):
    if hmodel.obsModel.DataAtomType == 'doc':
      LP = convertLPFromTokensToDocs(LP, Data)
    else:
      LP = convertLPFromDocsToTokens(LP, Data)

  ## Adjust "true" labels as specified by initname
  if initname == 'repeattruelabels':
    LP = expandLPWithDuplicates(LP, PRNG, nRepeatTrue)
  elif initname == 'truelabelsandempties':
    LP = expandLPWithEmpty(LP, 1)
  elif initname.count('junk'):
    LP = expandLPWithJunk(LP, 1, PRNG=PRNG)

  if hasattr(hmodel.allocModel, 'initLPFromResp'):
    LP = hmodel.allocModel.initLPFromResp(Data, LP)

  ## Perform global update step given these local params
  SS = hmodel.get_global_suff_stats(Data, LP)
  hmodel.update_global_params(SS)


def convertLPFromHardToSoft(LP, Data):
  ''' Transform array of hard assignment labels in Data into local param dict

      Returns
      ---------
      LP : local parameter dict, with fields 'resp'
  '''
  Z = LP['Z']
  uniqueLabels = np.unique(Z)
  Ktrue = len(uniqueLabels)
  resp = np.zeros((Data.nObs, Ktrue))
  for k in range(Ktrue):
    mask = Z == uniqueLabels[k]
    resp[mask,k] = 1.0
  return dict(resp=resp)

def expandLPWithEmpty(LP, nCol):
  ''' Create new LP by adding empty columns at the end
  '''
  resp = LP['resp']
  LP['resp'] = np.hstack([resp, np.zeros((resp.shape[0], nCol))])
  return LP

def expandLPWithJunk(LP, Kextra, PRNG=np.random.RandomState, fracJunk=0.01):
  ''' Create new LP by adding extra junk topics
  '''
  resp = LP['resp']
  N, K = resp.shape
  respNew = np.hstack([resp, np.zeros((N, Kextra))])
  Nextra = int(fracJunk * N)
  selectIDs = PRNG.choice(N, Nextra * Kextra).tolist()
  for k in xrange(Kextra):
    IDs_k = selectIDs[:Nextra]
    respNew[IDs_k, :K] = 0.01 / K
    respNew[IDs_k, K+k] = 1 - 0.01
    del selectIDs[:Nextra]
  return dict(resp=respNew)

def expandLPWithDuplicates(LP, PRNG, nRepeatTrue=2):
  ''' Create new LP by taking each existing component and duplicating it.

      Effectively there are nRepeatTrue "near-duplicates" of each comp,
        with each original member of comp k assigned to one of these duplicates

      Returns
      --------
      LP : local param dict, with field resp that has repeated comps
  '''
  resp = LP['resp']
  N, Ktrue = resp.shape
  rowIDs = PRNG.permutation(N)
  L = len(rowIDs)/nRepeatTrue
  bigResp = np.zeros((N, Ktrue*nRepeatTrue))
  curLoc = 0
  for r in range(nRepeatTrue):
    targetIDs = rowIDs[curLoc:curLoc+L]
    bigResp[targetIDs, r*Ktrue:(r+1)*Ktrue] = resp[targetIDs,:]
    curLoc += L
  LP['resp'] = bigResp
  return LP

def convertLPFromTokensToDocs(LP, Data):
  ''' Convert token-specific responsibilities into document-specific ones
  '''
  resp = LP['resp']
  N, K = resp.shape
  if N == Data.nDoc:
    return LP
  docResp = np.zeros((Data.nDoc, K))
  for d in xrange(Data.nDoc):
    respMatForDoc = resp[Data.doc_range[d]:Data.doc_range[d+1]]
    docResp[d,:] = np.mean( respMatForDoc, axis=0)
  LP['resp'] = docResp
  return LP

def convertLPFromDocsToTokens(LP, Data):
  ''' Convert doc-specific responsibilities into token-specific ones
  '''
  docResp = LP['resp']
  N, K = docResp.shape
  if N == Data.nUniqueToken:
    return LP
  tokResp = np.zeros((Data.nUniqueToken, K))
  for d in xrange(Data.nDoc):
    curDocResp = docResp[d]    
    start = Data.doc_range[d]
    stop = Data.doc_range[d+1]
    tokResp[start:stop,:] = curDocResp
  LP['resp'] = tokResp
  return LP
