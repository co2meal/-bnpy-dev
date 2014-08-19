'''
FromLP.py

Initialize global params of a bnpy model using a set of local parameters
'''
import numpy as np
import FromScratchMult

def init_global_params(hmodel, Data, initname='', initLP=None, **kwargs):
  ''' Initialize (in-place) the global params of the given hmodel
      using the true labels associated with the Data

      Args
      -------
      hmodel : bnpy model object to initialize
      Data   : bnpy Data object whose dimensions must match resulting hmodel
      initname : string name for the routine to use
                 'truelabels' or 'repeattruelabels'

      Returns
      --------
      None. hmodel global parameters updated in-place.
  '''
  if type(initLP) == dict:
    LP = initLP
  elif initname == 'contigblocksLP':
    LP = makeLP_ContigBlocks(Data, **kwargs)
  else:
    raise ValueError('Unrecognized initname: %s' % (initname))
  return initHModelFromLP(hmodel, Data, LP)    


def initHModelFromLP(hmodel, Data, LP):
  ''' Initialize provided bnpy HModel given data and local params.

      Executes summary step and global step.

      Returns
      --------
      None. hmodel global parameters updated in-place.
  '''
  if hasattr(hmodel.allocModel, 'initLPFromResp'):
    LP = hmodel.allocModel.initLPFromResp(Data, LP)
  SS = hmodel.get_global_suff_stats(Data, LP)
  hmodel.update_global_params(SS)


########################################################### ContigBlock init
###########################################################
def makeLP_ContigBlocks(Data, K=0, KperSeq=None, initNumSeq=None, **kwargs):
  ''' Create local parameters using a contiguous block hard segmentation.

      Divide chosen sequences up into KperSeq contiguous blocks (evenly sized),
      and assign each block to a separate unique hidden state.

      Returns
      -------
      LP : dict with fields
      * resp : 2D array, Natom x K
      data atom responsibility parametesr
  '''  
  if initNumSeq is None:
    initNumSeq = Data.nSeqs
  initNumSeq = np.minimum(initNumSeq, Data.nSeqs)

  if KperSeq is None:
    assert K > 0
    KperSeq = int(np.ceil(K / float(initNumSeq)))
    if KperSeq * initNumSeq > K:
      print 'WARNING: using initial K larger than suggested.'
    K = KperSeq * initNumSeq
  assert KperSeq > 0

  ## Select subset of all sequences to use for initialization
  if initNumSeq == Data.nSeqs:
    chosenSeqIDs = np.arange(initNumSeq)
  else:
    chosenSeqIDs = PRNG.choice(Data.nSeqs, initNumSeq, replace=False)

  ## Make hard segmentation at each chosen sequence
  resp = np.zeros((Data.nObs, K))
  jstart = 0
  for n in chosenSeqIDs:
    start = int(Data.seqInds[n])
    curT = Data.seqInds[n+1] - start

    ## Determine how long each block is for blocks 0, 1, ... KperSeq-1
    cumsumBlockSizes = calcBlockSizesForCurSeq(KperSeq, curT)
    for j in xrange(KperSeq):
      Tstart = start + cumsumBlockSizes[j]
      Tend = start + cumsumBlockSizes[j+1]
      resp[Tstart:Tend, jstart + j] = 1.0
    jstart = jstart + j + 1

  return dict(resp=resp)

def calcBlockSizesForCurSeq(KperSeq, curT):
  ''' Divide a sequence of length curT into KperSeq contig blocks

      Examples
      ---------
      >> calcBlockSizesForCurSeq(3, 20)
      [0, 7, 14, 20]

      Returns
      ---------
      c : 1D array, size KperSeq+1
      * block t indices are selected by c[t]:c[t+1]
  '''
  blockSizes = (curT // KperSeq) * np.ones(KperSeq)
  remMass = curT - np.sum(blockSizes)
  blockSizes[:remMass] += 1
  cumsumBlockSizes = np.cumsum(np.hstack([0,blockSizes]))
  return np.asarray(cumsumBlockSizes, dtype=np.int32)
