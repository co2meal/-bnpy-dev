'''
FromLP.py

Initialize global params of a bnpy model using a set of local parameters
'''
import numpy as np
import FromScratchMult
from FromTruth import convertLPFromHardToSoft
from bnpy.deletemove.DeleteMoveTarget import runDeleteMove_SingleSequence

import logging
Log = logging.getLogger('bnpy')
Log.setLevel(logging.DEBUG)

def init_global_params(hmodel, Data, initname='', initLP=None, 
                       **kwargs):
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
    return initHModelFromLP(hmodel, Data, initLP)

  elif initname == 'sacbLP':
    Log.info('Initialization: Sequential Allocation of Contig Blocks')
    SS = initSS_SeqAllocContigBlocks(Data, hmodel, **kwargs)
    hmodel.update_global_params(SS)
    return None

  elif initname == 'contigblocksLP':
    LP = makeLP_ContigBlocks(Data, **kwargs)
    return initHModelFromLP(hmodel, Data, LP)

  else:
    raise ValueError('Unrecognized initname: %s' % (initname))


def initHModelFromLP(hmodel, Data, LP):
  ''' Initialize provided bnpy HModel given data and local params.

      Executes summary step and global step.

      Returns
      --------
      None. hmodel global parameters updated in-place.
  '''
  if 'resp' not in LP:
    if 'Z' not in LP:
      raise ValueError("Bad LP. Require either 'resp' or 'Z' fields.")
    LP = convertLPFromHardToSoft(LP, Data)
  assert 'resp' in LP

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
    initNumSeq = Data.nDoc
  initNumSeq = np.minimum(initNumSeq, Data.nDoc)

  if KperSeq is None:
    assert K > 0
    KperSeq = int(np.ceil(K / float(initNumSeq)))
    if KperSeq * initNumSeq > K:
      print 'WARNING: using initial K larger than suggested.'
    K = KperSeq * initNumSeq
  assert KperSeq > 0

  ## Select subset of all sequences to use for initialization
  if initNumSeq == Data.nDoc:
    chosenSeqIDs = np.arange(initNumSeq)
  else:
    chosenSeqIDs = PRNG.choice(Data.nDoc, initNumSeq, replace=False)

  ## Make hard segmentation at each chosen sequence
  resp = np.zeros((Data.nObs, K))
  jstart = 0
  for n in chosenSeqIDs:
    start = int(Data.doc_range[n])
    curT = Data.doc_range[n+1] - start

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


def initSS_SeqAllocContigBlocks(Data, hmodel, **kwargs):
  if 'K' in kwargs and kwargs['K'] > 0:
    Kmax = int(kwargs['K'])
  else:
    Kmax = np.inf

  if 'seed' in kwargs:
    seed = int(kwargs['seed'])
  else:
    seed = 0
  # Traverse sequences in a random order
  PRNG = np.random.RandomState(seed)
  if hasattr(Data, 'nDoc'):
    randOrderIDs = range(Data.nDoc)
  else:
    randOrderIDs = [0]
  PRNG.shuffle(randOrderIDs)

  SS = None
  Ntotalsofar = 0
  for orderID, n in enumerate(randOrderIDs):
    Z_n, SS_n, SS = initSingleSeq_SeqAllocContigBlocks(
                                                 n, Data, hmodel,
                                                 SS=SS,
                                                 Kmax=Kmax,
                                                 mergeToSimplify=True,
                                                 returnSS=True,
                                                 **kwargs)
    Ntotalsofar += Z_n.size
    hmodel, SS, Result = runDeleteMove_SingleSequence(n, 
                                          Data, SS, SS_n, hmodel, 
                                          Kmax=Kmax, **kwargs)
    assert np.allclose(SS.N.sum(), Ntotalsofar)
    if orderID == len(randOrderIDs) - 1 \
       or (orderID+1) % 5 == 0 or orderID < 2:
        Log.info('  seq. %3d/%d | Ktotal=%d' 
                    % (orderID+1, len(randOrderIDs), SS.K))

  return SS

def initSingleSeq_SeqAllocContigBlocks(n, Data, hmodel,
                                       SS=None,
                                       seed=0,
                                       Kmax=np.inf,
                                       Kextra=5,
                                       allocFieldNames=None,
                                       initBlockLen=20,
                                       mergeToSimplify=False,
                                       **kwargs):
  ''' Initialize hard assignment state sequence Z for one single sequence
  '''
  if allocFieldNames is None:
    allocFieldNames = hmodel.allocModel.getSummaryFieldNames()
    allocFieldDims = hmodel.allocModel.getSummaryFieldDims()

  obsModel = hmodel.obsModel

  if hasattr(Data, 'doc_range'):
    start = Data.doc_range[n]
    stop = Data.doc_range[n+1]
    T = stop - start
  else:
    start = 0
    T = Data.nObs
  nBlocks = np.maximum(1, int(T // initBlockLen))

  # Loop over each contig block of data, and assign it en masse to one cluster
  Z = -1 * np.ones(T, dtype=np.int32)
  SSagg = SS
  tmpAllocFields = dict()
  if SSagg is None:
    kUID = 0
    Norig = 0
  else:
    kUID = SSagg.K
    Norig = SSagg.N.sum()
    for key in allocFieldNames:
      tmpAllocFields[key] = SSagg.removeField(key)

  ## We traverse the current sequence block by block,
  # Indices a,b denote the start and end of the current block *in this sequence*
  # SSactive denotes the most recent current stretch assigned to one comp
  # SSab denotes the current block 
  for blockID in xrange(nBlocks):
    if nBlocks == 1:
      a = 0
      b = T
    elif blockID == 0:
      # First block
      a = 0
      b = a + initBlockLen
    elif blockID == nBlocks - 1:
      # Final block
      a = b
      b = T
    else:
      # All interior blocks
      a = b
      b = a + initBlockLen

    SSab = obsModel.calcSummaryStatsForContigBlock(Data, a=start+a, b=start+b)
    if blockID == 0:
      Z[a:b] = kUID
      SSactive = SSab
      continue

    ELBOgap = obsModel.calcHardMergeGap_SpecificPairSS(SSactive, SSab)
    if (ELBOgap >= -0.000001):
      # Positive value means we prefer to assign block [a,b] to current state
      # So combine the current block into the active block 
      # and move on to the next block
      Z[a:b] = kUID
      SSactive += SSab
    else:
      # Negative value means we assign block [a,b] to a new state!
      Z[a:b] = kUID + 1
      SSagg, Z = updateAggSSWithFinishedCurrentBlock(SSagg, SSactive,   
                                       Z,
                                       obsModel,
                                       Kmax=Kmax+Kextra,
                                       mergeToSimplify=mergeToSimplify)
  
      # Create a new active block, starting at [a,b] 
      SSactive = SSab # make a soft copy / alias
      kUID = 1*SSagg.K

  # Final block needs to be recorded. 
  SSagg, Z = updateAggSSWithFinishedCurrentBlock(SSagg, SSactive, Z,
                                       obsModel,
                                       mergeToSimplify=mergeToSimplify)

  # Compute sequence-specific suff stats 
  # This includes allocmodel stats
  if hasattr(Data, 'nDoc'):
    Data_n = Data.select_subset_by_mask([n])
  else:
    Data_n = Data
  LP_n = convertLPFromHardToSoft(dict(Z=Z), Data_n, startIDsAt0=True,
                                                    Kmax=SSagg.K)
  LP_n = hmodel.allocModel.initLPFromResp(Data_n, LP_n)
  SS_n = hmodel.get_global_suff_stats(Data_n, LP_n)

  # Verify that our aggregate suff stats
  #  represent every single timestep in this sequence
  assert np.allclose(SSagg.N.sum() - Norig, Z.size)
  assert np.allclose(SS_n.N.sum(), Z.size)
  for ii, key in enumerate(allocFieldNames):
    dims = allocFieldDims[ii]
    if key in tmpAllocFields:
      arr, dims2 = tmpAllocFields[key]
      assert dims == dims2
      # Inflate with empties
      if len(dims) == 2 and dims[0] == 'K' and dims[1] == 'K':
        Kcur = arr.shape[0]
        Kextra = SSagg.K - Kcur 
        if Kextra > 0:
          arrBig = np.zeros((SSagg.K, SSagg.K))
          arrBig[:Kcur, :Kcur] = arr
          arr = arrBig
      elif len(dims) == 1 and dims[0] == 'K':
        Kcur = arr.size
        Kextra = SSagg.K - Kcur 
        if Kextra > 0:
          arr = np.append(arr, np.zeros(Kextra))

      arr += getattr(SS_n, key)

    else:
      arr = getattr(SS_n, key).copy()
    SSagg.setField(key, arr, dims)
  
  return Z, SS_n, SSagg

def updateAggSSWithFinishedCurrentBlock(SSagg, SScur, Z, obsModel,
                                        Kmax=np.inf,
                                        mergeToSimplify=False):
  ''' Store most recent tracked block into the aggregated bag of suff stats

      Will perform merges if requested to try to combine this recent block
      with any existing comps in the provided suff stat bag.

      Returns
      ----------
      SSagg : aggregated bag of suff stats for all tracked comps
      Z : state sequence for current time series
          possibly relabeled if a merge occurred.
  '''
  if SSagg is None:
    SSagg = SScur
  else:
    SSagg.insertComps(SScur)

  if mergeToSimplify and SSagg.K > 1:
    obsModel.update_global_params(SSagg)
    # Try to merge this recent block with all others
    mPairIDs = [(k, SSagg.K-1) for k in range(SSagg.K-1)]
    ELBOgaps = obsModel.calcHardMergeGap_SpecificPairs(SSagg, mPairIDs)
    bestID = np.argmax(ELBOgaps)
    if ELBOgaps[bestID] > 0 or SSagg.K > Kmax:
      kA, kB = mPairIDs[bestID]
      SSagg.mergeComps(kA, kB)
      # Reindex the state sequence Z
      Z[Z==kB] = kA
      Z[Z > kB] -= 1
  return SSagg, Z